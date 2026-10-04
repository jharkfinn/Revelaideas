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
| Tests | `nvfp4_unit_tests` and `tiny_parity` (random-weight model with the real attention geometry), `indexer_parity` and `selection_tie_stats` (real model). |

## Results so far (4 Oct 2026)

- **NVFP4 decode:** the expert module matches the transformers expert module with the same decoded weights (max difference 0). WikiText-103 validation, first 2,048 tokens: NLL 0.668 nats per token (perplexity 1.95). The greedy non-thinking answer to "What is the capital of France?" is `Paris`.
- **Scorer parity, random-weight model, GPU:** budgets of 64 and 2,048, lengths of 300, 2,100 and 4,096 tokens. Score difference is at most 1e-6. The masks are identical to the reference module. The logits are identical (difference 0) for the full forward pass and for the KV-cache continuation.
- **Scorer parity, real model, 3,000 WikiText tokens, all 12 QSA layers:** score difference is at most 3e-6. The masks are identical (0 rows differ). The RoPE and position-free parts sum back to the score.
- **Ties at the 512th block, real model:** on the 949 rows that must drop blocks, no row has a tie at the 512th score, and the 512th score is never 0. In all layers, 99.997% to 100% of blocks score above 0.
- **Speed (fast indexer, W4A16, no fused GDN kernels):**

  | Tokens | Seconds per forward pass | Tokens per second | Peak GPU memory |
  |---:|---:|---:|---:|
  | 2,048 | 4.0 | 513 | 73.7 GiB |
  | 4,096 | 4.9 | 842 | 74.2 GiB |
  | 8,192 | 6.2 | 1,320 | 75.7 GiB |
  | 10,400 | 7.0 | 1,496 | 76.6 GiB |

  About 3.3 s of each pass is a fixed cost: the per-expert decode loop over all 512 experts in 48 layers.

## How to run

1. Open the notebook on molab with a GPU attached. You can also run `marimo edit` on a machine with one GPU of at least 80 GB and about 60 GB of free RAM.
2. Download the checkpoint, about 133 GB: `hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --local-dir /root/models/qwen38-nvfp4`
3. Run the cells, then press **Load**. The load takes about 75 s from a local disk.

Requirements: `transformers==5.18.0` (it has `qwen4_exp`), `torch>=2.11` with CUDA, `safetensors`, `datasets`, and `polars`.
