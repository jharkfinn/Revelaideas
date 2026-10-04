# QSA indexer as a retrieval teacher (Qwen3.8-Flash-Next, NVFP4)

`qsa_teacher_qwen38_nvfp4.py` is a [marimo](https://marimo.io) notebook. It runs
[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) on one
96 GB GPU (molab RTX PRO 6000 Blackwell) and reads the Qwen Sparse Attention (QSA) indexer of the
12 softmax-attention layers. The goal is to test the indexer scores as a teacher signal for a
Revela-style dense retriever.

## What the notebook contains

| Part | Content |
|---|---|
| NVFP4 loader | `NVFP4Experts` keeps the ModelOpt NVFP4 routed experts packed on the GPU and decodes the experts that get tokens to bf16 in each forward pass (W4A16). `FP8RowTable` keeps the 320M x 160 FP8 PLE n-gram table in host RAM. `load_qwen38_nvfp4` builds the text-only `Qwen4ExpForCausalLM` on the meta device and fills it from the checkpoint. Transformers 5.18 cannot load this ModelOpt checkpoint by itself. |
| QSA scorer | `qsa_index_parts`, `qsa_block_scores` and `qsa_select_blocks` compute the indexer scores for all rows at once. `qsa_block_scores` can also return the dot products split into the RoPE part (dims 0 to 63) and the position-free part (dims 64 to 127). `fast_indexer_forward` replaces the per-query Python loop in `Qwen4ExpTextQSAIndexer.forward`. It works with and without a KV cache, and it falls back to the reference code when the mask has padding. |
| Attention side path | `attention_side_path` recomputes any QSA layer for chosen query rows: dense or sparse probabilities for all 24 heads, head outputs, and the elementwise sigmoid output gate. `indexer_selection_tokens` gives the token-level top-512 selection. |
| Phase 2 pilot | `build_rag_example` (chat format, non-thinking, passages padded or cut to exactly 160 tokens = 40 blocks), `extract_teacher_signals` (chunk-level teacher signals per layer and head), `start_pilot_job` (runs in a background thread), `pilot_score_table` and `pilot_cv_report` (gold-rank metrics; every head and layer choice is made on the other half of the questions). |
| Speed path | `nvfp4_dequant_kernel` (Triton), `grouped_experts_forward`, `use_fla_gdn`, `use_fast_gdn`, `use_compiled_hyper_connections`, `use_fused_glue`, `use_flex_attention` (block mask from `bool_mask_to_block_mask`), `use_sparse_attention` (Triton `qsa_sparse_attn_kernel` over `qsa_topk_block_ids`), `run_until` (early exit after a chosen layer), `use_fast_path`, `extract_teacher_batch` and `extract_teacher_multi` (batched prefix and suffix passes on an expanded KV cache), `start_pilot_job_fast`. |
| Phase 2b rerank eval | Pre-registration cell and `PHASE2B` config, `phase2b_bm25.py` and `phase2b_baselines.py` (separate processes), `build_rerank_example`, `rerank_signals_from_store` (temperature sweep), `start_rerank_job` (tiers smoke / dev / full), `phase2b_evaluate`, `phase2b_report`, `phase2b_lodo`, `phase2b_bias_report`. |
| Tests | `nvfp4_unit_tests` and `tiny_parity` (random-weight model with the real attention geometry), `indexer_parity` and `selection_tie_stats` (real model), the side-path check against the module output, and `profile_forward`. |

## Phase 1 results: loader and scorer (4 Oct 2026)

- **NVFP4 decode:** the expert module matches the transformers expert module with the same decoded weights (max difference 0). WikiText-103 validation, first 2,048 tokens: NLL 0.668 nats per token (perplexity 1.95). The greedy non-thinking answer to "What is the capital of France?" is `Paris`. A separate check against the bf16 checkpoint confirms the format: re-quantizing the bf16 expert reproduces 100% of the packed bytes, with the low nibble holding the even index and the dequant computed as `e2m1 * weight_scale * weight_scale_2`.
- **Scorer parity, random-weight model, GPU:** budgets of 64 and 2,048, lengths of 300, 2,100 and 4,096 tokens. Score difference is at most 1e-6. The masks are identical to the reference module. The logits are identical (difference 0) for the full forward pass and for the KV-cache continuation.
- **Scorer parity, real model, 3,000 WikiText tokens, all 12 QSA layers:** score difference is at most 3e-6. The masks are identical (0 rows differ). The RoPE and position-free parts sum back to the score.
- **Side path, real model:** the recomputed sparse attention output matches the module output to within about 1% of its largest value (bf16 attention noise).
- **Ties at the 512th block, real model:** on the 949 rows that must drop blocks, no row has a tie at the 512th score, and the 512th score is never 0. In all layers, 99.997% to 100% of blocks score above 0.
- **Speed (fast indexer, W4A16, no fused GDN kernels):**

  | Tokens | Seconds per pass | Experts | GDN | QSA attention | Peak GPU memory |
  |---:|---:|---:|---:|---:|---:|
  | 2,600 | 4.3 | 3.87 s | 0.27 s | 0.06 s | 73.7 GiB |
  | 10,400 | 7.0 | 4.65 s | 1.16 s | 0.53 s | 76.6 GiB |

  The NVFP4 expert path takes 67% to 89% of the time. It decodes all 512 experts in 48 layers in every pass and runs a Python loop over the experts.

## Speed path (fast enough to iterate)

`use_fast_path()` switches on five patches. Each one can be turned off on its own to compare with the reference path:

| Patch | What it does | 10,260-token forward |
|---|---|---:|
| Reference | per-expert loop, PyTorch NVFP4 decode, PyTorch GDN, eager hyper-connections, SDPA | 7.0 s |
| `use_grouped_experts` | Triton NVFP4 decode at about 1 TB/s into a reusable buffer, decode only the experts that get tokens, `torch._grouped_mm`, one weighted `bmm` to combine the 10 expert outputs per token | 3.6 s |
| `use_fla_gdn` | `flash-linear-attention` kernels for the Gated DeltaNet layers (rebinds the functions in the loaded module) | 2.79 s |
| `use_compiled_hyper_connections` | `torch.compile` of the gated-residual module, one graph shared by all instances | 2.47 s |
| `use_flex_attention` | FlexAttention with a block mask built from the causal and indexer mask, for queries of 1,024 rows or more | 2.28 s |

The pilot extraction (`extract_teacher_multi`) adds three more savings:
- the question and the null question share one cached prefix pass;
- all suffixes run as one right-padded batch, which is safe because attention is causal;
- 3 questions run per pass (peak 87 GiB; 4 questions need 91 GiB).

| Pilot | Reference path | Fast path |
|---|---:|---:|
| One K = 64 question (question + null) | 27.3 s | 2.27 s |
| K = 16, 96 questions | 864 s | 94 s |
| K = 64, 64 questions | 1,750 s | 145 s |

**Parity.**
- The grouped experts reproduce the per-expert loop exactly (logit KL 0).
- The fla kernels, the compiled hyper-connections, FlexAttention and the KV-cache continuation each change the bf16 numeric path. Each step shifts the WikiText logits by a KL of about 0.015 to 0.02, with top-1 agreement of about 96%. Per GDN call, fla differs from the reference by 3.4e-3 relative error, about twice the reference's own bf16 error against fp32.
- On the teacher signal the fast pilot agrees with the reference pilot (`results/pilot_fast_cv_report_*.csv`):
  - the Spearman correlation of the indexer chunk masses is 0.993 (K = 16) and 0.996 (K = 64);
  - layers 31 and 35 give the same gold rank on 81% to 91% of questions;
  - every cross-validated MRR and nDCG is within 0.02, except the native top-512 share at K = 64 (0.42 → 0.34);
  - the indexer-vs-dense conclusion does not change.

### Speed path, part 2: long rerank prompts (25K tokens)

The rerank prompts of Phase 2b have 100 documents, about 25K tokens. At this length the pilot fast path took 5.9 s per query (query + null query). A profile showed these costs:
- the per-expert matmul loop that `torch._grouped_mm` runs on this GPU (sm_120);
- FlexAttention. The indexer picks 4-token blocks separately for each row, so nearly every 128 x 128 tile is partial, and the kernel does close to dense causal work;
- dense 25K x 25K boolean masks (650 MB each), built and combined several times per QSA layer;
- the layout copies of the Gated DeltaNet forward.

| Step (`results/speed_steps_25k.csv`) | s per query | Parity with the step before |
|---|---:|---|
| Pilot fast path, all 48 layers | 5.9 | |
| `run_until(..., last_layer=35)`: stop after layer 35. The layer-31 indexer and the QRHead heads (layers 27, 31, 35) need no later layer | 4.4 | signals of layers 3 to 35 identical (max difference 2e-6) |
| `use_sparse_attention`: the indexer returns its top-512 block ids, and a Triton kernel attends to the selected blocks and the tail tokens. One program per (row, KV head), with the 12 query heads of the KV head as the matmul rows. No T x T mask exists | 3.6 | kernel against masked SDPA: 0.2% relative error. Logits KL 0.011 against FlexAttention, teacher Spearman 0.994 or higher |
| `use_fast_gdn`: grouped value heads go straight to the fla chunk kernel (no `repeat_interleave`), a compiled 4-tap causal conv1d + SiLU on the [B, T, C] layout, and a compiled gated RMSNorm | 3.1 | logits KL 0.013, teacher Spearman 0.995 or higher |
| `use_fused_glue`: compiled hyper-connection injection, MoE combine (gather + weighted sum) and SwiGLU | 2.95 | logits KL 0.012, teacher Spearman 0.987 or higher |

**End to end on the smoke tier.** DL19 + DL20 (97 queries) took 157 s on the current fast path. Against the records of the pilot fast path (all layers, FlexAttention):
- primary nDCG@10: 0.608 → 0.603 (DL19) and 0.566 → 0.568 (DL20);
- QRHead-16: 0.635 → 0.638 and 0.590 → 0.592;
- per-query Spearman of the calibrated layer-31 scores: mean 0.993, minimum 0.972.

**Total drift of the fast path.** Reference path against the full fast path, WikiText 10,240 tokens, all 48 layers: logits KL 0.013, top-1 agreement 96.6%, NLL 1.121 (reference) and 1.112 (fast). The rounding differences of the separate steps do not add up. A 10,240-token forward takes 6.55 s on the reference path and 1.80 s on the fast path.

**Notes.**
- The cached continuation differs from one full pass by about 12% relative error in the last hidden states (logits KL about 0.02). The reference path shows the same, so this is bf16 noise and not a cache bug.
- The fla `causal_conv1d` kernel autotunes again for each new sequence-length bucket (about 5 s each time). It is not used.
- The remaining costs at 25K tokens, layers up to 35:
  - the per-expert matmul loop: about 0.4 s;
  - the NVFP4 decode into bf16: 0.2 s;
  - the sparse attention kernel: 0.2 s;
  - the dense projections: about 0.4 s.

  A grouped GEMM with the NVFP4 decode fused in, or FP4 tensor cores (W4A4, as vLLM does), would cut most of the first two.

Lessons for long runs on molab:
- Run long jobs in a background thread (`start_pilot_job*`, `start_rerank_job`). When an HTTP request to the kernel is cancelled, marimo interrupts the running cell.
- `set_ui_value` on a run button already runs the cell. A second explicit `run_cell` runs it again and starts a second job. `start_rerank_job` now refuses to start while a job thread with the same name is alive.
- In scratchpad code, keep large tensors inside functions, because loop variables and failed calls can keep GPU memory alive.
- Do not give Triton kernels names that start with `_`. marimo renames such names, and Triton then cannot find the function.
- `pyserini` 2.4 installs torch 2.14, numpy 2.5 and a CUDA 13 stack. In the molab venv these shadowed the system torch 2.11. Run pyserini in a separate process (`phase2b_bm25.py`, with `JAVA_HOME` set), and remove the extra torch, triton, numpy, `nvidia-*` and `cuda-*` packages from the venv after the install.

## Phase 2 pilot: does the indexer point at the gold passage?

Data: Natural Questions dev (Tevatron/wikipedia-nq), 1 gold passage plus K - 1 BM25 hard negatives. The gold slot cycles over the K slots. Rows: the question tokens, and the rows that predict the teacher-forced answer. Null calibration subtracts the chunk mass from a pass where the question is "N/A".

Fair comparison on **question rows** (`results/pilot_cv_report_*.csv`). Every head or layer choice is made on the other half of the questions:

| Teacher | K = 16: MRR | K = 16: nDCG@10 | K = 64: MRR | K = 64: nDCG@10 |
|---|---:|---:|---:|---:|
| Indexer, best layer (CV), null-calibrated | 0.757 | 0.818 | **0.738** | **0.794** |
| Indexer, best layer (CV), raw | 0.738 | 0.800 | 0.660 | 0.729 |
| Indexer, z-scored mean of the 12 layers, null-calibrated | 0.720 | 0.786 | 0.616 | 0.692 |
| Native top-512 share, best layer (CV) | 0.517 | 0.610 | 0.421 | 0.518 |
| Dense heads, top 16 of 288 (CV) | **0.810** | **0.855** | 0.721 | 0.777 |
| Sparse heads, top 16 of 288 (CV) | 0.805 | 0.851 | 0.731 | 0.784 |
| Random order | 0.211 | 0.284 | 0.074 | 0.071 |

The rows below are the paired bootstrap difference, indexer (CV, calibrated) minus dense top 16 (CV), with 95% intervals:

- **K = 16** (96 questions, about 75% of blocks kept): MRR −0.053 [−0.089, −0.021], nDCG@10 −0.036 [−0.064, −0.010].
- **K = 64** (64 questions, 10.4K tokens, about 20% of blocks kept): MRR +0.017 [−0.027, +0.061], nDCG@10 +0.017 [−0.017, +0.052].

Other findings:

- **Answer rows inflate the score.** In NQ, the gold passage contains the answer string and the BM25 negatives are filtered to exclude it. So the teacher-forced answer rows can find the gold by string match (answer-row MRR about 0.95 at K = 16). Only question rows test retrieval from the question.
- **Depth profile.** The signal is near random at layer 3, peaks at layers 31 and 35, then falls toward layer 47.
- **Null calibration is required.** It adds 0.1 to 0.2 MRR at mid-depth layers for K = 64.
- **Position bias goes toward the first passages, not the recent ones.** Raw mass on non-gold chunks falls with the slot index. The ratio of (slope x (K − 1)) to the gold margin is −0.27 at layer 31 for K = 64. Calibration brings it to −0.07, below the 10% gate at layers 31 and 35. Layer 47 stays at −0.20.
- **The RoPE / position-free split gives no free de-biased teacher.** The position-free half alone ranks worse than the full score at mid and late layers. The RoPE half alone is weak.
- **Paper-scale softmax** (softmax over the unscaled score of Eq. 15) is mixed. It is slightly better at K = 16 and worse at K = 64.
- **Output-gate weighting** does not change the ranks at the chunk level.
- **Caveats.** These are small samples (96 and 64 questions). The layer choice is cross-validated, but the variants were explored on the same data. About 20% of passages were cut at 159 tokens. The loader runs W4A16, while vLLM serves this checkpoint as W4A4.

## Phase 2b: standard rerank evaluation (BM25 top 100, nDCG@10)

**Design.** The full pre-registration is in the notebook cell "Phase 2b".
- **Data.** TREC DL19 and DL20 passage, and the BEIR test sets of TREC-COVID, NFCorpus, SciFact and FiQA.
- **First stage.** Pyserini BM25 flat, top 100. Every system reads the same documents, cut at 256 Qwen tokens.
- **Teacher prompt.** The documents in a fixed random order per query, then `Query: ...`. The signals are read on the query tokens and calibrated against the null query `N/A`.
- **Primary teacher.** The layer-31 indexer, code-scale softmax at tau = 1, block mass summed per document, null-calibrated. The layer and the 16 QRHead heads come from the NQ K = 64 pilot, so these test sets are held out.
- **Metric and test.** nDCG@10 with trec_eval semantics (pytrec_eval). Paired bootstrap over queries, macro mean over datasets.

**Amendment, made after DL19, DL20 and 8 TREC-COVID queries were known.** Speed is now the main goal, so:
- all passes stop after layer 35;
- ICR-288 (all 288 heads) is not computed;
- one baseline (MiniLM) stays as a sanity check;
- runs use tiers: smoke (DL19 + DL20, 97 queries), dev (+ TREC-COVID and 100 fixed random queries each of NFCorpus, SciFact and FiQA, 447 queries) and full (1,418 queries).

**Sanity check of the eval.**
- BM25 reproduces the Pyserini regression values on the full sets: DL19 0.5058, DL20 0.4796, TREC-COVID 0.5947, NFCorpus 0.3218, SciFact 0.6789.
- MiniLM-L6 reproduces the BEIR-paper reranking numbers to within about 0.016: TREC-COVID 0.741, NFCorpus 0.350, SciFact 0.682.

**Dev-tier results** (447 queries; `results/phase2b_dev_*.csv`; nDCG@10):

| System | DL19 | DL20 | TREC-COVID | NFCorpus | SciFact | FiQA | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle order of the top 100 (ceiling) | 0.892 | 0.871 | 0.975 | 0.543 | 0.949 | 0.635 | 0.811 |
| QRHead-16, calibrated | 0.635 | 0.590 | 0.770 | 0.354 | 0.790 | 0.498 | **0.606** |
| MiniLM-L6 cross-encoder (supervised on MS MARCO) | 0.727 | 0.675 | 0.741 | 0.346 | 0.683 | 0.403 | 0.596 |
| **Indexer L31, tau = 1, calibrated (primary)** | 0.608 | 0.566 | 0.749 | 0.348 | 0.791 | 0.479 | **0.590** |
| Indexer, layer and tau chosen leave-one-dataset-out | 0.608 | 0.578 | 0.744 | 0.353 | 0.776 | 0.469 | 0.588 |
| Indexer, z-scored mean of layers 3 to 35 | 0.569 | 0.520 | 0.749 | 0.346 | 0.770 | 0.449 | 0.567 |
| QRHead-16, raw | 0.469 | 0.455 | 0.679 | 0.328 | 0.784 | 0.484 | 0.533 |
| Indexer L31, raw | 0.420 | 0.370 | 0.650 | 0.300 | 0.784 | 0.459 | 0.497 |
| BM25 | 0.506 | 0.480 | 0.595 | 0.320 | 0.690 | 0.293 | 0.480 |
| Random order (mean of 20) | 0.215 | 0.160 | 0.400 | 0.161 | 0.051 | 0.038 | 0.171 |

The rows below are the paired bootstrap of the primary teacher minus each system, macro mean, with 95% intervals:
- **H1, against BM25:** +0.110 [+0.087, +0.133]. It is better on every set. The NFCorpus interval touches 0: +0.028 [−0.001, +0.058].
- **H2, against QRHead-16:** −0.016 [−0.023, −0.009]. By the pre-registered rule the indexer is "worse", because the upper bound is below 0. The lower bound is just outside the 0.02 margin. Per set, the difference is −0.028 (DL19), −0.023 (DL20), −0.021 (TREC-COVID), −0.019 (FiQA), and −0.007 and +0.001 on NFCorpus and SciFact.
- **Against MiniLM:** −0.006 [−0.024, +0.013]. The indexer is worse on DL19 and DL20, which are in the MiniLM training domain (−0.12 and −0.11). It is better on SciFact (+0.108) and FiQA (+0.076).
- **Against the leave-one-dataset-out pick:** +0.002 [−0.003, +0.007]. Layer 31 wins on every held-out split. tau = 1.414 wins 5 of 6, but this gains only about 0.004.

**Bias check** (`results/phase2b_dev_bias.csv`). This is the Spearman correlation on non-relevant documents. The slot is random, so a slot correlation is pure position bias.
- **Raw indexer scores** favour early slots (rho −0.24 to −0.31) and long documents (rho up to +0.41 on DL).
- **After null calibration:** |rho_slot| ≤ 0.11 and |rho_len| ≤ 0.10.
- **Calibrated QRHead-16** over-corrects on SciFact (rho_slot +0.23).

**What this says.**
- The indexer alone, one layer and no head choice, is a zero-shot reranker on par with a supervised MiniLM cross-encoder outside MS MARCO. It is 0.016 nDCG behind the best 16 attention heads.
- As a teacher, the indexer has two practical gains: its signal is one distribution per row that the model already computes, and it was trained with a KL loss to be a ranking distribution.
- It is not a better signal than the heads.

**Caveats.**
- The dev tier is a subset, so the full tier is still to run.
- The amendment came after a small look at the data.
- The records come from three versions of the fast path: DL from full depth with FlexAttention, the rest from exit at 35. The teacher Spearman between paths is 0.99 or higher.
- ICR-288 is not in the comparison.
- One prompt template, not tuned.
- Documents are cut at 256 tokens.
- The model runs W4A16, while vLLM serves this checkpoint as W4A4.

## Corrections to the earlier plan

- The tech report's residual-path analysis is about the gated-residual streams of a 20-layer probe model. It does not show that the softmax-attention layers are the main long-range readers.
- MRCR 8-needle at 512K: QSA 40.53, full attention 30.66.
- The paper's indexer score (Eq. 15) has no 1/sqrt(128) factor, and the KL uses softmax(I). HF and the vLLM AMD path divide by sqrt(128). The vLLM NVIDIA kernels do not. This does not change top-k selection.
- The attention output gate is elementwise (24 heads x 256 channels), not one scalar per head.

## How to run

1. Open the notebook on molab with a GPU attached. You can also run `marimo edit` on a machine with one GPU of at least 80 GB and about 60 GB of free RAM.
2. Download the checkpoint, about 133 GB: `hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --local-dir /root/models/qwen38-nvfp4`
3. Run the cells, then press **Load**. The load takes about 75 s from a local disk.
4. Press **Run pilot**. It runs in a background thread for about 30 minutes, so a dropped browser or HTTP request cannot interrupt it. Then run the loader cell below it.
5. Phase 2b:
   - Copy `phase2b_bm25.py` and `phase2b_baselines.py` to `PHASE2B["cand_dir"]` and press **Build BM25 candidates and baseline runs**. This needs pyserini and Java 21, takes about 5 minutes, and runs in its own process.
   - Choose a tier and press **Run Phase 2b teacher job**. On the current fast path the smoke tier took 157 s (measured, 97 queries). For the dev tier I estimate about 17 minutes, based on 2.95 s per 25K-token query (not measured end to end). Records that already exist are skipped.
   - Press **Refresh Phase 2b results**.

Requirements: `transformers==5.18.0` (it has `qwen4_exp`), `torch>=2.11` with CUDA (Triton 3.6), `flash-linear-attention==0.5.2`, `safetensors`, `datasets`, `polars`, and `pytrec-eval-terrier`. For the Phase 2b first stage: `pyserini==2.4.0` with a Java 21 JRE (`JAVA_HOME`) in the environment that runs `phase2b_bm25.py`. See the pyserini note under "Lessons" before you install it next to the notebook.
