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

Requirements: `transformers==5.18.0` (it has `qwen4_exp`), `torch>=2.11` with CUDA, `safetensors`, `datasets`, and `polars`.
