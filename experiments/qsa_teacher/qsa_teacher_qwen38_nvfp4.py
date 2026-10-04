# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "flash-linear-attention==0.5.2",
#     "numpy==2.4.6",
#     "safetensors==0.8.0",
#     "transformers==5.18.0",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    import json
    import math
    import os
    import time

    import altair as alt
    import marimo as mo
    import polars as pl
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    import transformers.models.qwen4_exp.modeling_qwen4_exp as qm
    from transformers import AutoTokenizer, Qwen4ExpConfig, Qwen4ExpForCausalLM, Qwen4ExpTextConfig


@app.cell
def _():
    mo.md(r"""
    # QSA indexer as a retrieval teacher: Qwen3.8-Flash-Next (NVFP4)

    This notebook runs `nvidia/Qwen3.8-Flash-Next-NVFP4` on one RTX PRO 6000 (96 GB) and reads the
    Qwen Sparse Attention (QSA) indexer of its 12 softmax-attention layers.

    **Indexer score** for query row $t$, key block $b$, layer $\ell$:

    $$s^{\ell}_{t,b} = \frac{1}{\sqrt{d_I}} \sum_{h=1}^{H} \mathrm{ReLU}\big(\langle q^{\ell}_{t,h}, \bar k^{\ell}_{b} \rangle\big), \qquad b < \lfloor (t+1)/r \rfloor$$

    - $H = 4$ indexer query heads, $d_I = 128$ indexer head dim, $r = 4$ tokens per block.
    - $q^{\ell}_{t,h}$ = RoPE(RMSNorm($W_Q x_t$)) on dims 0..63; $\bar k^{\ell}_b$ = RoPE(RMSNorm(mean of $W_K x_i$ over the 4 tokens of block $b$)) at the block start position.
    - The model attends to the top 512 blocks plus the 0 to 3 tokens of the incomplete last block.

    **Loader.** Routed experts stay NVFP4 on the GPU and are decoded to bf16 per forward (W4A16).
    The PLE n-gram table (320M rows x 160 dims = 51.2B values, FP8) stays in host RAM. All other weights are bf16 as stored.
    """)
    return


@app.cell
def _():
    MODEL_DIR = "/root/models/qwen38-nvfp4"
    DEVICE = "cuda"
    return DEVICE, MODEL_DIR


@app.function
def dequant_nvfp4(w_u8, scale_u8, scale2_rows, out_dtype=torch.bfloat16):
    """Dequantize ModelOpt NVFP4 weights.

    w_u8:        [..., N, K/2] uint8, two E2M1 codes per byte (low nibble = even k).
    scale_u8:    [..., N, K/16] uint8 holding FP8 E4M3 block scales (one per 16 k).
    scale2_rows: [..., N] fp32 global scale (weight_scale_2) broadcast per row.
    Returns w[..., n, k] = e2m1(code) * fp8(scale[n, k // 16]) * scale2[n].
    """
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=w_u8.device,
    )
    codes = torch.stack([w_u8 & 0x0F, w_u8 >> 4], dim=-1).flatten(-2).long()
    vals = lut[codes]
    block = scale_u8.view(torch.float8_e4m3fn).float() * scale2_rows.float()[..., None]
    vals = vals.unflatten(-1, (-1, 16)) * block[..., None]
    return vals.flatten(-2).to(out_dtype)


@app.function
def quant_nvfp4_reference(w, block=16):
    """Reference NVFP4 weight quantizer (for round-trip tests only).

    Returns (w_u8, scale_u8, scale2) such that dequant_nvfp4 reproduces w up to FP4/FP8 rounding.
    """
    w = w.float()
    amax = w.abs().amax().clamp_min(1e-12)
    scale2 = amax / (6.0 * 448.0)
    wb = w.unflatten(-1, (-1, block))
    s = (wb.abs().amax(-1) / 6.0 / scale2).clamp(max=448.0).to(torch.float8_e4m3fn)
    denom = (s.float() * scale2)[..., None].clamp_min(1e-30)
    x = (wb / denom).clamp(-6, 6)
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device)
    idx = (x.abs()[..., None] - grid).abs().argmin(-1)
    code = idx + 8 * (x < 0).long()
    code = code.flatten(-2)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    return packed, s.view(torch.uint8), scale2


@app.class_definition
class NVFP4Experts(torch.nn.Module):
    """Drop-in for Qwen4ExpTextExperts that stores ModelOpt NVFP4 packed weights.

    Each forward dequantizes the experts that receive tokens (chunk by chunk) to bf16 and runs
    the same SwiGLU expert math as the reference module (W4A16). Gate and up rows are stacked
    as [gate; up] like the fused reference `gate_up_proj`.
    """

    def __init__(self, num_experts, hidden_dim, intermediate_dim, device, chunk=16):
        super().__init__()
        E, H, I = num_experts, hidden_dim, intermediate_dim
        self.num_experts, self.hidden_dim, self.intermediate_dim = E, H, I
        self.chunk = chunk
        self.register_buffer("gu_q", torch.empty(E, 2 * I, H // 2, dtype=torch.uint8, device=device))
        self.register_buffer("gu_s", torch.empty(E, 2 * I, H // 16, dtype=torch.uint8, device=device))
        self.register_buffer("gu_g", torch.empty(E, 2, dtype=torch.float32, device=device))
        self.register_buffer("gu_in", torch.empty(E, 2, dtype=torch.float32, device=device))
        self.register_buffer("d_q", torch.empty(E, H, I // 2, dtype=torch.uint8, device=device))
        self.register_buffer("d_s", torch.empty(E, H, I // 16, dtype=torch.uint8, device=device))
        self.register_buffer("d_g", torch.empty(E, dtype=torch.float32, device=device))
        self.register_buffer("d_in", torch.empty(E, dtype=torch.float32, device=device))

    def dense_gate_up(self, ids, dtype=torch.bfloat16):
        I = self.intermediate_dim
        g = self.gu_g[ids]
        rows = torch.cat([g[:, :1].expand(-1, I), g[:, 1:].expand(-1, I)], dim=1)
        return dequant_nvfp4(self.gu_q[ids], self.gu_s[ids], rows, dtype)

    def dense_down(self, ids, dtype=torch.bfloat16):
        rows = self.d_g[ids][:, None].expand(-1, self.hidden_dim)
        return dequant_nvfp4(self.d_q[ids], self.d_s[ids], rows, dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        T, K = top_k_index.shape
        out = torch.zeros(T, self.hidden_dim, dtype=torch.float32, device=hidden_states.device)
        flat = top_k_index.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.num_experts)
        ends = counts.cumsum(0)
        starts = ends - counts
        hit = torch.nonzero(counts).flatten()
        tok_sorted = order // K
        w_sorted = top_k_weights.reshape(-1)[order]
        hit_l, starts_l, ends_l = hit.tolist(), starts.tolist(), ends.tolist()
        for c0 in range(0, len(hit_l), self.chunk):
            ids = hit[c0 : c0 + self.chunk]
            w_gu = self.dense_gate_up(ids, hidden_states.dtype)
            w_d = self.dense_down(ids, hidden_states.dtype)
            for j, e in enumerate(hit_l[c0 : c0 + self.chunk]):
                s, t = starts_l[e], ends_l[e]
                tok = tok_sorted[s:t]
                gate, up = F.linear(hidden_states[tok], w_gu[j]).chunk(2, dim=-1)
                y = F.linear(F.silu(gate) * up, w_d[j])
                out.index_add_(0, tok, y.float() * w_sorted[s:t, None].float())
            del w_gu, w_d
        return out.to(hidden_states.dtype)


@app.class_definition
class FP8RowTable(torch.nn.Module):
    """CPU-resident FP8 E4M3 embedding table with one per-tensor scale.

    Replaces the PLE `ngram_embedding` nn.Embedding. Rows are gathered on CPU as uint8, moved to
    the GPU, then decoded to bf16 and multiplied by the scale.
    """

    def __init__(self, num_rows, dim, out_device):
        super().__init__()
        self.table_u8 = torch.empty(num_rows, dim, dtype=torch.uint8)
        self.scale = torch.ones(1, dtype=torch.bfloat16)
        self.out_device = torch.device(out_device)
        # Qwen4ExpTextNGramEmbedding reads `.weight.device` to choose where to gather rows.
        self.weight = torch.empty(0)

    def forward(self, ids):
        rows = self.table_u8[ids.cpu()].to(self.out_device, non_blocking=True)
        vals = rows.view(torch.float8_e4m3fn).to(torch.bfloat16)
        return vals * self.scale.to(self.out_device)


@app.function
def ckpt_key(name):
    return "model.language_model." + name[len("model.") :] if name.startswith("model.") else name


@app.function
def set_module_tensor(model, name, tensor, as_param):
    mod_name, _, attr = name.rpartition(".")
    mod = model.get_submodule(mod_name) if mod_name else model
    if as_param:
        mod._parameters[attr] = torch.nn.Parameter(tensor, requires_grad=False)
    else:
        mod._buffers[attr] = tensor


@app.function
def load_qwen38_nvfp4(model_dir, device="cuda", log=print):
    """Build text-only Qwen4ExpForCausalLM from the ModelOpt NVFP4 checkpoint.

    Routed experts stay NVFP4 on the GPU (NVFP4Experts); the PLE n-gram table stays FP8 on the CPU
    (FP8RowTable); every other tensor is loaded as stored (bf16 / int64) onto `device`.
    """
    t0 = time.time()
    config = Qwen4ExpConfig.from_pretrained(model_dir).text_config
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(config)
    E, H, I = config.num_experts, config.hidden_size, config.moe_intermediate_size
    for layer in model.model.layers:
        layer.mlp.experts = NVFP4Experts(E, H, I, device)
    ple_owner = None
    for layer in model.model.layers:
        if layer.ple is not None:
            ple_owner = layer.ple.ple_embedding
            n_rows, dim = ple_owner.ngram_embedding.weight.shape
            ple_owner.ngram_embedding = FP8RowTable(n_rows, dim, device)

    weight_map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(key, dev="cpu"):
        fname = weight_map[key]
        if (fname, dev) not in handles:
            handles[(fname, dev)] = safe_open(os.path.join(model_dir, fname), framework="pt", device=dev)
        return handles[(fname, dev)].get_tensor(key)

    # 1) plain bf16 / int64 tensors
    skip = ("mlp.experts.", "ple_embedding.ngram_embedding.")
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    rotary_names = {n for n in buffers if n.startswith("model.rotary_emb.")}
    missing = []
    for name, tensor in list(params.items()) + list(buffers.items()):
        if any(s in name for s in skip) or name in rotary_names or tensor.device.type != "meta":
            continue
        key = ckpt_key(name)
        if key not in weight_map:
            missing.append(name)
            continue
        t = get(key, str(device))
        if tuple(t.shape) != tuple(tensor.shape):
            raise ValueError(f"shape mismatch {name}: ckpt {tuple(t.shape)} vs model {tuple(tensor.shape)}")
        set_module_tensor(model, name, t, name in params)
    if missing:
        raise KeyError(f"tensors missing from checkpoint: {missing[:20]} (+{max(0, len(missing) - 20)} more)")
    model.model.rotary_emb = qm.Qwen4ExpTextRotaryEmbedding(config).to(device)
    log(f"dense tensors loaded in {time.time() - t0:.1f}s")

    # 2) NVFP4 routed experts
    for li, layer in enumerate(model.model.layers):
        ex = layer.mlp.experts
        pre = f"model.language_model.layers.{li}.mlp.experts"
        gu_q = torch.empty(ex.gu_q.shape, dtype=torch.uint8)
        gu_s = torch.empty(ex.gu_s.shape, dtype=torch.uint8)
        d_q = torch.empty(ex.d_q.shape, dtype=torch.uint8)
        d_s = torch.empty(ex.d_s.shape, dtype=torch.uint8)
        gu_g = torch.empty(E, 2)
        gu_in = torch.empty(E, 2)
        d_g = torch.empty(E)
        d_in = torch.empty(E)
        for e in range(E):
            for j, proj in enumerate(("gate_proj", "up_proj")):
                k = f"{pre}.{e}.{proj}"
                rows = slice(j * I, (j + 1) * I)
                gu_q[e, rows] = get(k + ".weight")
                gu_s[e, rows] = get(k + ".weight_scale").view(torch.uint8)
                gu_g[e, j] = get(k + ".weight_scale_2")
                gu_in[e, j] = get(k + ".input_scale")
            k = f"{pre}.{e}.down_proj"
            d_q[e] = get(k + ".weight")
            d_s[e] = get(k + ".weight_scale").view(torch.uint8)
            d_g[e] = get(k + ".weight_scale_2")
            d_in[e] = get(k + ".input_scale")
        for buf, src in ((ex.gu_q, gu_q), (ex.gu_s, gu_s), (ex.d_q, d_q), (ex.d_s, d_s),
                         (ex.gu_g, gu_g), (ex.gu_in, gu_in), (ex.d_g, d_g), (ex.d_in, d_in)):
            buf.copy_(src)
        if li % 8 == 7:
            log(f"experts loaded through layer {li} ({time.time() - t0:.0f}s)")

    # 3) FP8 n-gram table (CPU)
    if ple_owner is not None:
        table = ple_owner.ngram_embedding
        prefix = None
        for li, layer in enumerate(model.model.layers):
            if layer.ple is not None:
                prefix = f"model.language_model.layers.{li}.ple.ple_embedding.ngram_embedding"
        off = 0
        for s in range(config.split_ngram_parts):
            shard = get(f"{prefix}.shard_{s}.weight").view(torch.uint8)
            table.table_u8[off : off + shard.shape[0]] = shard
            off += shard.shape[0]
        if off != table.table_u8.shape[0]:
            raise ValueError(f"n-gram rows {off} != expected {table.table_u8.shape[0]}")
        table.scale = get(f"{prefix}.weight_scale").to(torch.bfloat16)
        log(f"n-gram table loaded ({off} rows, scale {table.scale.item():.6g}) at {time.time() - t0:.0f}s")

    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == "meta"]
    if left:
        raise RuntimeError(f"meta tensors left after load: {left[:20]}")
    model.eval()
    log(f"model ready in {time.time() - t0:.0f}s")
    return model


@app.function
def qsa_index_parts(indexer, x, cos, sin):
    """Indexer queries and pooled block keys for one unpadded sequence.

    x: [T, D] attention-module input; cos/sin: [T, rot] for positions 0..T-1.
    Returns q [T, H, d] and kbar [T // r, d], both in model dtype, computed with the same op order
    and dtypes as Qwen4ExpTextQSAIndexer.forward.
    """
    T = x.shape[0]
    r, d, Hq = indexer.compress_ratio, indexer.index_head_dim, indexer.index_n_heads
    q, k = torch.split(indexer.index_qk_proj(x), [Hq * d, d], dim=-1)
    q = indexer.q_layernorm(q.reshape(T, Hq, d))
    q = qm.apply_rotary_pos_emb(q, cos=cos, sin=sin, unsqueeze_dim=1)
    nb = T // r
    kbar = k[: nb * r].reshape(nb, r, d).float().mean(dim=1).to(k.dtype)
    kbar = indexer.k_layernorm(kbar)
    starts = torch.arange(nb, device=x.device) * r
    kbar = qm.apply_rotary_pos_emb(kbar.unsqueeze(1), cos=cos[starts], sin=sin[starts]).squeeze(1)
    return q, kbar


@app.function
def qsa_block_scores(q_rows, kbar, positions, r, rope_dim=None):
    """Indexer block scores for query rows.

    q_rows: [R, H, d]; kbar: [nb, d]; positions: [R] absolute positions of the rows.
    Returns dict with `score` [R, nb] fp32 (-inf where block b is not complete and visible, i.e.
    b >= (t + 1) // r), and when rope_dim is given the per-head dot products split into the RoPE
    part (dims < rope_dim) and the position-free part.
    """
    d = q_rows.shape[-1]
    qf, kf = q_rows.float(), kbar.float()
    dots = torch.einsum("rhd,bd->rbh", qf, kf)
    score = torch.relu(dots).sum(-1) / math.sqrt(d)
    nb = kbar.shape[0]
    valid = torch.arange(nb, device=kbar.device)[None, :] < ((positions + 1) // r)[:, None]
    out = {"score": score.masked_fill(~valid, float("-inf")), "valid": valid}
    if rope_dim is not None:
        out["dots_rope"] = torch.einsum("rhd,bd->rbh", qf[..., :rope_dim], kf[:, :rope_dim])
        out["dots_nope"] = torch.einsum("rhd,bd->rbh", qf[..., rope_dim:], kf[:, rope_dim:])
    return out


@app.function
def qsa_select_blocks(score, k_blocks):
    """Top-k blocks per row among valid ones. Returns bool [R, nb]."""
    k = min(k_blocks, score.shape[-1])
    top = score.topk(k, dim=-1)
    sel = torch.zeros_like(score, dtype=torch.bool)
    sel.scatter_(-1, top.indices, top.values > float("-inf"))
    return sel


@app.function
def reference_scores_loop(indexer, x, cos, sin, rows):
    """Literal copy of the reference per-query loop (Qwen4ExpTextQSAIndexer.forward) that returns
    the raw block scores instead of the selection, for an unpadded sequence."""
    T = x.shape[0]
    d, r = indexer.index_head_dim, indexer.compress_ratio
    qk = indexer.index_qk_proj(x[None])
    q, token_k = torch.split(qk, [indexer.index_n_heads * d, d], dim=-1)
    q, raw_keys = q.reshape(1, T, -1, d), token_k.reshape(1, T, -1, d).squeeze(2)
    q = indexer.q_layernorm(q)
    q = qm.apply_rotary_pos_emb(q, cos=cos[None], sin=sin[None], unsqueeze_dim=2)
    out = {}
    for t in rows:
        vis = torch.arange(t + 1, device=x.device)
        nblk = vis.shape[-1] // r
        if nblk == 0:
            out[t] = torch.empty(0, device=x.device)
            continue
        bti = vis[: nblk * r].view(nblk, r)
        kg = raw_keys[0].index_select(0, bti.flatten()).view(*bti.shape, d)
        pooled = indexer.k_layernorm(kg.float().mean(dim=1).to(raw_keys.dtype))
        gs = bti[:, 0]
        bk = qm.apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cos.index_select(0, gs), sin=sin.index_select(0, gs)).squeeze(1)
        sc = torch.matmul(q[0, t].float(), bk.float().transpose(-1, -2)).transpose(-1, -2)
        out[t] = torch.relu(sc).sum(dim=-1) / math.sqrt(d)
    return out


@app.function
def fast_indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
    """Vectorized Qwen4ExpTextQSAIndexer.forward for unpadded causal inputs (with or without cache).

    Falls back to the reference loop when the mask is not plain causal. Returns the same additive
    or boolean mask as the reference.
    """
    B, Tq, _ = hidden_states.shape
    Tk = attention_mask.shape[-1]
    vis = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    causal = torch.arange(Tk, device=vis.device)[None, :] <= (torch.arange(Tq, device=vis.device) + Tk - Tq)[:, None]
    if not torch.equal(vis[:, 0], causal.expand(B, -1, -1)):
        return qm.Qwen4ExpTextQSAIndexer._reference_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)
    r, d, Hq = self.compress_ratio, self.index_head_dim, self.index_n_heads
    full_cos, full_sin = position_embeddings
    cur_cos, cur_sin = full_cos[:, -Tq:, :], full_sin[:, -Tq:, :]
    q, token_k = torch.split(self.index_qk_proj(hidden_states), [Hq * d, d], dim=-1)
    q = self.q_layernorm(q.reshape(B, Tq, Hq, d))
    q = qm.apply_rotary_pos_emb(q, cos=cur_cos, sin=cur_sin, unsqueeze_dim=2)
    raw_keys = token_k.reshape(B, Tq, d)
    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)
    nb = Tk // r
    positions = torch.arange(Tk - Tq, Tk, device=hidden_states.device)
    mask = torch.zeros(B, Tq, Tk, dtype=torch.bool, device=hidden_states.device)
    for b in range(B):
        if nb > 0:
            kbar = raw_keys[b, : nb * r].reshape(nb, r, d).float().mean(dim=1).to(raw_keys.dtype)
            kbar = self.k_layernorm(kbar)
            starts = torch.arange(nb, device=hidden_states.device) * r
            kbar = qm.apply_rotary_pos_emb(
                kbar.unsqueeze(1), cos=full_cos[b, starts], sin=full_sin[b, starts]
            ).squeeze(1)
            for c0 in range(0, Tq, 2048):
                sl = slice(c0, min(Tq, c0 + 2048))
                sc = qsa_block_scores(q[b, sl], kbar, positions[sl], r)["score"]
                sel = qsa_select_blocks(sc, self.block_topk)
                mask[b, sl, : nb * r] = sel.repeat_interleave(r, dim=-1)
        tail_start = ((positions + 1) // r) * r
        cols = torch.arange(Tk, device=hidden_states.device)
        mask[b] |= (cols[None, :] >= tail_start[:, None]) & (cols[None, :] <= positions[:, None])
    mask = mask.unsqueeze(1)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        mask = torch.where(mask, attention_mask.new_zeros(()), min_dtype)
    return mask


@app.function
def use_fast_indexer(enabled=True):
    """Swap the vectorized indexer into Qwen4ExpTextQSAIndexer.forward (or restore the reference)."""
    cls = qm.Qwen4ExpTextQSAIndexer
    if not hasattr(cls, "_reference_forward"):
        cls._reference_forward = cls.forward
    cls.forward = fast_indexer_forward if enabled else cls._reference_forward
    return enabled


@app.function
def reference_indexer_forward(indexer, *args):
    """Call the original transformers indexer forward even when the fast one is installed."""
    cls = qm.Qwen4ExpTextQSAIndexer
    return getattr(cls, "_reference_forward", cls.forward)(indexer, *args)


@app.function
def tiny_qwen4_config(budget):
    """Random-weight Qwen4-Exp config with the real attention/indexer geometry (head dim 256, 64 RoPE
    dims, M-RoPE sections, indexer 4 x 128, 4-token blocks) and small everything else."""
    return Qwen4ExpTextConfig(
        hidden_size=256, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2, head_dim=256,
        layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
        vocab_size=512, indexer_budget=budget, indexer_compress_ratio=4, indexer_head_dim=128,
        indexer_n_heads=4, indexer_kv_heads=1, hc_count=4, hc_lowrank=16, ple_layer_ids=[2],
        ple_embed_dim=256, ngram_vocab_size_base=1000, heads_per_ngram=8, ngram_size=3,
        output_gate_type="sigmoid", eos_token_id=1, bos_token_id=1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                         "mrope_section": [11, 11, 10], "mrope_interleaved": True},
    )


@app.function
def capture_attention_inputs(model):
    """Forward pre-hooks on every QSA attention module. store[layer] = (x [B,T,D], (cos, sin) [B,T,rot])."""
    store, hooks = {}, []
    for li, layer in enumerate(model.model.layers):
        if layer.layer_type != "linear_attention":
            def pre(mod, args, kwargs, li=li):
                store[li] = (args[0].detach(), tuple(a.detach() for a in args[1]))
            hooks.append(layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    return store, hooks


@app.function
def indexer_parity(indexer, x, cos, sin, rows):
    """Compare the vectorized scorer and the fast mask against the reference for one layer.

    x: [1, T, D]; cos/sin: [1, T, rot]. Returns max score difference on `rows`, whether the RoPE /
    no-position split sums back to the score, the number of rows whose selection differs, and
    whether every difference is an exact tie at the 512th (k-th) score.
    """
    dev = x.device
    q, kbar = qsa_index_parts(indexer, x[0], cos[0], sin[0])
    pos = torch.tensor(rows, device=dev)
    rope_dim = cos.shape[-1]
    vec = qsa_block_scores(q[pos], kbar, pos, indexer.compress_ratio, rope_dim=rope_dim)
    ref = reference_scores_loop(indexer, x[0], cos[0], sin[0], rows)
    md = 0.0
    for i, t in enumerate(rows):
        nv = (t + 1) // indexer.compress_ratio
        if nv:
            md = max(md, (vec["score"][i, :nv] - ref[t]).abs().max().item())
    resum = torch.relu(vec["dots_rope"] + vec["dots_nope"]).sum(-1) / math.sqrt(q.shape[-1])
    split_ok = torch.allclose(resum.masked_fill(~vec["valid"], float("-inf")), vec["score"], atol=1e-3, rtol=1e-4)
    T = x.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()[None, None]
    mref = reference_indexer_forward(indexer, x, (cos, sin), causal, None)
    mfast = fast_indexer_forward(indexer, x, (cos, sin), causal, None)
    diff_rows = (mref != mfast).any(-1)[0, 0].nonzero().flatten().tolist()
    ties_only, max_gap = True, 0.0
    if diff_rows:
        check = diff_rows[:: max(1, len(diff_rows) // 32)]
        refd = reference_scores_loop(indexer, x[0], cos[0], sin[0], check)
        for t in check:
            s = refd[t]
            kth = s.topk(min(indexer.block_topk, s.numel())).values[-1]
            dcols = (mref[0, 0, t] != mfast[0, 0, t]).nonzero().flatten()
            in_blocks = bool((dcols < (t + 1) // indexer.compress_ratio * indexer.compress_ratio).all())
            gap = (s[torch.unique(dcols // indexer.compress_ratio)] - kth).abs().max().item()
            max_gap = max(max_gap, gap)
            ties_only &= in_blocks and gap <= 1e-5
    return dict(score_maxdiff=md, split_ok=bool(split_ok), mask_diff_rows=len(diff_rows),
                ties_only=ties_only, max_gap_to_kth=max_gap)


@app.function
def tiny_parity(budget, T, rows, device, seed=0):
    """Random-weight model: vectorized scorer / fast indexer vs reference, incl. full logits and the
    KV-cache continuation path. Returns one row per QSA layer."""
    torch.manual_seed(seed)
    model = Qwen4ExpForCausalLM(tiny_qwen4_config(budget)).to(device).to(torch.bfloat16).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "layernorm" in n or n.endswith("norm.weight"):
                p.normal_(0, 0.3)
    ids = torch.randint(2, 500, (1, T), device=device)
    store, hooks = capture_attention_inputs(model)
    use_fast_indexer(False)
    with torch.no_grad():
        ref_logits = model(ids).logits.float()
    for h in hooks:
        h.remove()
    use_fast_indexer(True)
    with torch.no_grad():
        fast_logits = model(ids).logits.float()
    split = T - 37
    cont = {}
    for fast in (False, True):
        use_fast_indexer(fast)
        with torch.no_grad():
            o1 = model(ids[:, :split], use_cache=True)
            cont[fast] = model(ids[:, split:], past_key_values=o1.past_key_values, use_cache=True).logits.float()
    use_fast_indexer(False)
    out = []
    for li, (x, (cos, sin)) in store.items():
        with torch.no_grad():
            r = indexer_parity(model.model.layers[li].self_attn.indexer, x, cos, sin, rows)
        out.append(dict(case=f"budget={budget} T={T}", layer=li, **r,
                        logit_maxdiff=(ref_logits - fast_logits).abs().max().item(),
                        cache_fast_vs_ref=(cont[True] - cont[False]).abs().max().item(),
                        cache_vs_full=(cont[False] - ref_logits[:, split:]).abs().max().item()))
    return out


@app.function
def nvfp4_unit_tests(device, seed=1):
    """(1) Round trip through a reference NVFP4 quantizer; (2) NVFP4Experts vs the transformers
    experts module loaded with the same dequantized weights."""
    torch.manual_seed(seed)
    w = torch.randn(64, 256, device=device) * 0.02
    q, s, g = quant_nvfp4_reference(w)
    rel = ((dequant_nvfp4(q, s, g.expand(64), torch.float32) - w).norm() / w.norm()).item()
    E, H, I = 8, 256, 64
    ex = NVFP4Experts(E, H, I, device, chunk=3)
    gu = torch.empty(E, 2 * I, H, device=device)
    dn = torch.empty(E, H, I, device=device)
    for e in range(E):
        for j in range(2):
            wq, ws, wg = quant_nvfp4_reference(torch.randn(I, H, device=device) * 0.05)
            ex.gu_q[e, j * I:(j + 1) * I], ex.gu_s[e, j * I:(j + 1) * I], ex.gu_g[e, j] = wq, ws, wg
            gu[e, j * I:(j + 1) * I] = dequant_nvfp4(wq, ws, wg.expand(I), torch.float32)
        wq, ws, wg = quant_nvfp4_reference(torch.randn(H, I, device=device) * 0.05)
        ex.d_q[e], ex.d_s[e], ex.d_g[e] = wq, ws, wg
        dn[e] = dequant_nvfp4(wq, ws, wg.expand(H), torch.float32)
    cfg = tiny_qwen4_config(64)
    cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size = E, H, I
    cfg._experts_implementation = "eager"
    ref = qm.Qwen4ExpTextExperts(cfg).to(device)
    with torch.no_grad():
        ref.gate_up_proj.copy_(gu)
        ref.down_proj.copy_(dn)
        x = torch.randn(50, H, device=device)
        p = torch.randn(50, E, device=device).softmax(-1)
        tv, ti = p.topk(2, -1)
        tv = tv / tv.sum(-1, keepdim=True)
        a, b = ex(x, ti, tv), ref(x, ti, tv)
    return dict(roundtrip_rel_err=rel, experts_maxdiff=(a - b).abs().max().item(), experts_absmax=b.abs().max().item())


@app.cell
def _(DEVICE):
    nvfp4_unit = nvfp4_unit_tests(DEVICE)
    nvfp4_unit
    return


@app.cell
def _(DEVICE):
    _t0 = time.time()
    tiny_parity_table = pl.DataFrame(
        tiny_parity(64, 300, [0, 3, 4, 5, 63, 64, 65, 66, 67, 255, 299], DEVICE)
        + tiny_parity(2048, 2100, [2047, 2050, 2051, 2052, 2099], DEVICE)
        + tiny_parity(2048, 4096, [2051, 2052, 3000, 4095], DEVICE)
    )
    print(f"tiny parity took {time.time() - _t0:.1f}s")
    tiny_parity_table
    return


@app.cell
def _():
    load_button = mo.ui.run_button(label="Load Qwen3.8-Flash-Next NVFP4 (~78 GB GPU + 51 GB host RAM)")
    load_button
    return (load_button,)


@app.cell
def _(DEVICE, MODEL_DIR, load_button):
    mo.stop(not load_button.value, mo.md("Press **Load** to build the model. Reruns of upstream cells unload it."))
    def _log(msg):
        print(msg, flush=True)
        with open("/root/models/load.log", "a") as _f:
            _f.write(msg + "\n")
    with mo.status.spinner(title="Loading NVFP4 checkpoint ..."):
        model = load_qwen38_nvfp4(MODEL_DIR, DEVICE, log=_log)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    use_fast_indexer(True)
    mo.md(f"Loaded. GPU memory in use: **{torch.cuda.memory_allocated() / 2**30:.1f} GiB**")
    return model, tokenizer


@app.function
def load_wikitext_ids(tokenizer, n_tokens, split="validation"):
    """First `n_tokens` tokens of WikiText-103 (raw) as one [1, T] tensor."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    text = "".join(ds["text"][:4000])
    ids = tokenizer(text, return_tensors="pt").input_ids
    if ids.shape[1] < n_tokens:
        raise ValueError(f"only {ids.shape[1]} tokens available")
    return ids[:, :n_tokens]


@app.function
@torch.no_grad()
def lm_nll(model, ids, chunk=1024):
    """Mean next-token NLL over ids [1, T]; logits computed in chunks to bound memory."""
    h = model.model(ids.to(model.lm_head.weight.device)).last_hidden_state
    total, n = 0.0, 0
    for s in range(0, h.shape[1] - 1, chunk):
        e = min(h.shape[1] - 1, s + chunk)
        logits = model.lm_head(h[:, s:e]).float()
        tgt = ids[:, s + 1 : e + 1].to(logits.device)
        total += F.cross_entropy(logits[0], tgt[0], reduction="sum").item()
        n += e - s
    return total / n


@app.function
def selection_tie_stats(indexer, x, cos, sin, chunk=1024):
    """For rows where the indexer must drop blocks ((t+1)//r > 512): how often the 512th score ties
    the 513th, how often the 512th score is exactly 0 (ReLU floor), and how many blocks score > 0."""
    q, kbar = qsa_index_parts(indexer, x[0], cos[0], sin[0])
    T, r, k = x.shape[1], indexer.compress_ratio, indexer.block_topk
    pos = torch.arange(T, device=x.device)
    rows = pos[(pos + 1) // r > k]
    if rows.numel() == 0:
        return dict(active_rows=0)
    tie, zero, npos = [], [], []
    for c0 in range(0, rows.numel(), chunk):
        rc = rows[c0 : c0 + chunk]
        sc = qsa_block_scores(q[rc], kbar, rc, r)["score"]
        top = sc.topk(k + 1, dim=-1).values
        tie.append(top[:, k - 1] == top[:, k])
        zero.append(top[:, k - 1] == 0)
        npos.append((sc > 0).sum(-1).float() / ((rc + 1) // r).float())
    tie, zero, npos = torch.cat(tie), torch.cat(zero), torch.cat(npos)
    return dict(active_rows=int(rows.numel()), frac_tie_at_kth=tie.float().mean().item(),
                frac_kth_zero=zero.float().mean().item(), mean_frac_blocks_positive=npos.mean().item())


@app.cell
def _(DEVICE, MODEL_DIR, model, tokenizer):
    mo.stop("model" not in globals(), mo.md("Load the model first."))
    wiki_ids = load_wikitext_ids(tokenizer, 12000)
    _t0 = time.time()
    sanity_nll = lm_nll(model, wiki_ids[:, :2048])
    _dt = time.time() - _t0
    _chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France? Answer with one word."}],
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True,
    )["input_ids"].to(DEVICE)
    from transformers import GenerationConfig as _GC
    with torch.no_grad():
        _gen = model.generate(_chat, max_new_tokens=16, do_sample=False,
                              generation_config=_GC.from_pretrained(MODEL_DIR, do_sample=False))
    sanity_answer = tokenizer.decode(_gen[0, _chat.shape[1]:])
    mo.md(f"""
    **WikiText-103 val, first 2,048 tokens:** NLL = {sanity_nll:.3f} nats/token, perplexity = {math.exp(sanity_nll):.2f} ({_dt:.1f}s forward)

    **Greedy answer (non-thinking):** `{sanity_answer!r}`
    """)
    return (wiki_ids,)


@app.cell
def _(DEVICE, model, wiki_ids):
    mo.stop("model" not in globals(), mo.md("Load the model first."))
    _ids = wiki_ids[:, :3000].to(DEVICE)
    _store, _hooks = capture_attention_inputs(model)
    with torch.no_grad():
        model.model(_ids)
    for _h in _hooks:
        _h.remove()
    _rows = [0, 3, 4, 2047, 2050, 2051, 2052, 2053, 2500, 2999]
    _res = []
    _t0 = time.time()
    for _li, (_x, (_cos, _sin)) in _store.items():
        with torch.no_grad():
            _idx = model.model.layers[_li].self_attn.indexer
            _r = indexer_parity(_idx, _x, _cos, _sin, _rows)
            _r.update(selection_tie_stats(_idx, _x, _cos, _sin))
        _res.append(dict(layer=_li, **_r))
    del _store
    real_parity_table = pl.DataFrame(_res)
    print(f"real-model parity on T=3000 took {time.time() - _t0:.0f}s")
    real_parity_table
    return


@app.cell
def _(DEVICE, model, wiki_ids):
    mo.stop("model" not in globals(), mo.md("Load the model first."))
    def _time_forward(T, reps=1):
        _x = wiki_ids[:, :T].to(DEVICE)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        _t = time.time()
        for _ in range(reps):
            with torch.no_grad():
                model.model(_x)
        torch.cuda.synchronize()
        return dict(T=T, seconds=(time.time() - _t) / reps, tokens_per_s=T * reps / (time.time() - _t),
                    peak_GiB=torch.cuda.max_memory_allocated() / 2**30)
    use_fast_indexer(True)
    forward_timing = pl.DataFrame([_time_forward(T) for T in (2048, 4096, 8192, 10400)])
    use_fast_indexer(False)
    _ref = _time_forward(2048)
    use_fast_indexer(True)
    mo.vstack([mo.md(f"Reference indexer loop at T=2048: **{_ref['seconds']:.1f}s** per forward"), forward_timing])
    return


@app.cell
def _():
    mo.md(r"""
    ## Phase 2 pilot: does the indexer point at the gold passage?

    Natural Questions dev (Tevatron/wikipedia-nq): 1 gold passage + 15 BM25 hard negatives per question,
    each passage padded or cut to exactly 160 tokens (40 indexer blocks). The gold position cycles over
    the 16 slots. The prompt uses the chat template in non-thinking mode; the gold answer is teacher-forced.

    Query rows: the question tokens ($G_q$) and the rows that predict each answer token ($G_a$).
    For layer $\ell$ and chunk $j$ (token span $P_j$):

    $$\pi^{\ell}_t(b) = \mathrm{softmax}_{b \in V_t}\big(s^{\ell}_{t,b}\big), \qquad m^{\ell}_j = \frac{1}{|G|}\sum_{t \in G}\ \sum_{b \in P_j} \pi^{\ell}_t(b), \qquad c^{\ell}_j = m^{\ell}_j - m^{\ell,\mathrm{null}}_j$$

    - $V_t$ = complete blocks visible to row $t$; $G$ = a row group; $m^{\ell,\mathrm{null}}_j$ = the same mass with the question replaced by "N/A".
    - Dense-head teachers use the attention probability mass of each of the 12 x 24 = 288 heads on $P_j$, with all causal keys (dense) or with the indexer's own selection (sparse).
    """)
    return


@app.function
def build_rag_example(tokenizer, question, passages, answer, chunk_len=160, r=4, max_answer=32):
    """Chat-format RAG prompt with every passage exactly `chunk_len` tokens and block-aligned.

    Returns ids [1, T], chunk token spans, question rows, and the rows that predict each answer token.
    """
    def enc(s):
        return tokenizer(s, add_special_tokens=False).input_ids
    nl = enc("\n")[0]
    pre = enc("<|im_start|>user\nRead the passages, then answer the question in a few words.\n\n")
    pre = pre + [nl] * (-len(pre) % r)
    spans, body = [], []
    for i, p in enumerate(passages):
        c = enc(f"[{i + 1}] {p['title']}\n{p['text']}")[: chunk_len - 1]
        c = c + [nl] * (chunk_len - len(c))
        start = len(pre) + i * chunk_len
        spans.append((start, start + chunk_len))
        body += c
    q = enc(f"Question: {question}")
    mid = enc("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    a = enc(answer)[:max_answer]
    ids = pre + body + q + mid + a
    q0 = len(pre) + len(body)
    a0 = q0 + len(q) + len(mid)
    return dict(
        ids=torch.tensor([ids]),
        spans=spans,
        q_rows=torch.arange(q0, q0 + len(q)),
        a_rows=torch.arange(a0 - 1, a0 - 1 + len(a)),
    )


@app.function
def attention_side_path(attn, x, cos, sin, rows, sel_tokens=None):
    """Recompute one QSA attention layer for query rows from its input.

    attn: Qwen4ExpTextAttention; x: [T, D] layer input (one unpadded sequence); cos/sin: [T, rot];
    rows: LongTensor [R] absolute positions; sel_tokens: optional bool [R, T] indexer selection
    (None = dense causal attention).
    Returns probs [R, Hh, T] fp32, head outputs o [R, Hh, dh] (before the gate), gate sigma(g)
    [R, Hh, dh], and the module output y [R, D] (gated heads through o_proj).
    """
    T = x.shape[0]
    dh = attn.head_dim
    qg = attn.q_proj(x[rows]).view(rows.numel(), -1, dh * 2)
    query, gate = torch.chunk(qg, 2, dim=-1)
    q = attn.q_norm(query)
    k = attn.k_norm(attn.k_proj(x).view(T, -1, dh))
    v = attn.v_proj(x).view(T, -1, dh)
    q = qm.apply_rotary_pos_emb(q, cos=cos[rows], sin=sin[rows], unsqueeze_dim=1)
    k = qm.apply_rotary_pos_emb(k, cos=cos, sin=sin, unsqueeze_dim=1)
    rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    logits = torch.einsum("rhd,thd->rht", q.float(), k.float()) * attn.scaling
    allowed = torch.arange(T, device=x.device)[None, :] <= rows[:, None]
    if sel_tokens is not None:
        allowed = allowed & sel_tokens
    logits = logits.masked_fill(~allowed[:, None, :], float("-inf"))
    probs = logits.softmax(-1)
    o = torch.einsum("rht,thd->rhd", probs.to(v.dtype), v)
    g = torch.sigmoid(gate.float())
    y = attn.o_proj((o.float() * g).to(x.dtype).reshape(rows.numel(), -1))
    return dict(probs=probs, o=o, gate=g, y=y)


@app.function
def indexer_selection_tokens(indexer, x, cos, sin, rows):
    """Token-level selection mask [R, T] of the indexer (top blocks + 0..3 tail tokens) and scores."""
    T = x.shape[0]
    r = indexer.compress_ratio
    q, kbar = qsa_index_parts(indexer, x, cos, sin)
    sc = qsa_block_scores(q[rows], kbar, rows, r, rope_dim=cos.shape[-1])
    sel_b = qsa_select_blocks(sc["score"], indexer.block_topk)
    nb = kbar.shape[0]
    tok = torch.zeros(rows.numel(), T, dtype=torch.bool, device=x.device)
    tok[:, : nb * r] = sel_b.repeat_interleave(r, dim=-1)
    cols = torch.arange(T, device=x.device)
    tail0 = ((rows + 1) // r) * r
    tok |= (cols[None, :] >= tail0[:, None]) & (cols[None, :] <= rows[:, None])
    return tok, sc, sel_b


@app.function
def chunk_mass(token_weights, spans):
    """token_weights [..., T]; spans list of (start, end) token ranges -> [..., K] summed weight."""
    return torch.stack([token_weights[..., s:e].sum(-1) for s, e in spans], dim=-1)


@app.function
@torch.no_grad()
def extract_teacher_signals(model, ex):
    """One forward pass of a build_rag_example output; chunk-level teacher signals per QSA layer.

    Row groups G: 0 = question rows, 1 = answer rows, 2 = both. Returns CPU float tensors:
      idx   [L, G, 8, K]: indexer softmax mass over blocks for (full score, RoPE part, position-free
                          part) at the code scale s = I / sqrt(d) and at the paper scale I (the
                          distribution the indexer was trained on), then the mean and the max block
                          score in the chunk. Order: mass, mass_rope, mass_nope, mass_paper,
                          mass_rope_paper, mass_nope_paper, mean_score, max_score
      sel   [L, G, K]:    share of the chunk's blocks inside the top-512 selection
      dense [L, H, G, K]: dense attention mass per head; sparse: same under the indexer selection
      gate  [L, H, G]:    mean sigmoid output gate; gnorm: norm of the gated head output (sparse)
    """
    device = model.lm_head.weight.device
    store, hooks = capture_attention_inputs(model)
    try:
        model.model(ex["ids"].to(device))
    finally:
        for h in hooks:
            h.remove()
    spans = ex["spans"]
    nq, na = ex["q_rows"].numel(), ex["a_rows"].numel()
    rows = torch.cat([ex["q_rows"], ex["a_rows"]]).to(device)
    groups = [slice(0, nq), slice(nq, nq + na), slice(0, nq + na)]
    out = {k: [] for k in ("idx", "sel", "dense", "sparse", "gate", "gnorm")}
    for li in sorted(store):
        x, (cos, sin) = store[li]
        x, cos, sin = x[0], cos[0], sin[0]
        attn = model.model.layers[li].self_attn
        r, d = attn.indexer.compress_ratio, attn.indexer.index_head_dim
        tok, sc, selb = indexer_selection_tokens(attn.indexer, x, cos, sin, rows)
        valid = sc["valid"]
        neg = float("-inf")
        s_rope = (torch.relu(sc["dots_rope"]).sum(-1) / math.sqrt(d)).masked_fill(~valid, neg)
        s_nope = (torch.relu(sc["dots_nope"]).sum(-1) / math.sqrt(d)).masked_fill(~valid, neg)
        full = sc["score"]
        masses = [torch.stack([(v * scale).softmax(-1)[:, s // r : e // r].sum(-1) for s, e in spans], -1)
                  for scale in (1.0, math.sqrt(d)) for v in (full, s_rope, s_nope)]
        finite = full.masked_fill(~valid, 0.0)
        mean_sc = torch.stack([finite[:, s // r : e // r].mean(-1) for s, e in spans], -1)
        max_sc = torch.stack([full[:, s // r : e // r].amax(-1) for s, e in spans], -1)
        idx = torch.stack(masses + [mean_sc, max_sc], 1)
        sel = torch.stack([selb[:, s // r : e // r].float().mean(-1) for s, e in spans], -1)
        sp = attention_side_path(attn, x, cos, sin, rows, tok)
        de = attention_side_path(attn, x, cos, sin, rows, None)
        dense, sparse = chunk_mass(de["probs"], spans), chunk_mass(sp["probs"], spans)
        gate = sp["gate"].mean(-1)
        gnorm = (sp["o"].float() * sp["gate"]).norm(dim=-1)
        out["idx"].append(torch.stack([idx[g].mean(0) for g in groups]))
        out["sel"].append(torch.stack([sel[g].mean(0) for g in groups]))
        out["dense"].append(torch.stack([dense[g].mean(0) for g in groups], 1))
        out["sparse"].append(torch.stack([sparse[g].mean(0) for g in groups], 1))
        out["gate"].append(torch.stack([gate[g].mean(0) for g in groups], 1))
        out["gnorm"].append(torch.stack([gnorm[g].mean(0) for g in groups], 1))
    del store
    return {k: torch.stack(v).float().cpu() for k, v in out.items()}


@app.cell
def _(DEVICE, model, wiki_ids):
    mo.stop("model" not in globals(), mo.md("Load the model first."))
    _ids = wiki_ids[:, :3000].to(DEVICE)
    _ins, _outs, _hooks = {}, {}, []
    for _li, _layer in enumerate(model.model.layers):
        if _layer.layer_type != "linear_attention":
            _hooks.append(_layer.self_attn.register_forward_pre_hook(
                lambda m, a, k, li=_li: _ins.__setitem__(li, (a[0].detach(), a[1])), with_kwargs=True))
            _hooks.append(_layer.self_attn.register_forward_hook(
                lambda m, a, o, li=_li: _outs.__setitem__(li, o[0].detach())))
    with torch.no_grad():
        model.model(_ids)
    for _h in _hooks:
        _h.remove()
    _rows = torch.tensor([5, 1000, 2050, 2051, 2052, 2600, 2999], device=DEVICE)
    _res = []
    with torch.no_grad():
        for _li in sorted(_ins):
            _x, (_cos, _sin) = _ins[_li]
            _attn = model.model.layers[_li].self_attn
            _tok, _, _ = indexer_selection_tokens(_attn.indexer, _x[0], _cos[0], _sin[0], _rows)
            _sp = attention_side_path(_attn, _x[0], _cos[0], _sin[0], _rows, _tok)
            _de = attention_side_path(_attn, _x[0], _cos[0], _sin[0], _rows, None)
            _ref = _outs[_li][0, _rows].float()
            _res.append(dict(layer=_li, sparse_vs_module=(_sp["y"].float() - _ref).abs().max().item(),
                             dense_vs_module=(_de["y"].float() - _ref).abs().max().item(),
                             module_absmax=_ref.abs().max().item()))
    del _ins, _outs
    side_path_check = pl.DataFrame(_res)
    side_path_check
    return


@app.function
def nq_pilot_examples(n, k):
    """First n NQ-dev questions with >= k-1 BM25 negatives; gold slot cycles over 0..k-1."""
    from datasets import load_dataset
    ds = load_dataset(
        "parquet",
        data_files={"dev": "hf://datasets/Tevatron/wikipedia-nq@refs%2Fconvert%2Fparquet/default/dev/*.parquet"},
        split="dev",
    )
    out = []
    for row in ds:
        if not row["positive_passages"] or not row["answers"] or len(row["negative_passages"]) < k - 1:
            continue
        gold = len(out) % k
        negs = row["negative_passages"][: k - 1]
        out.append(dict(question=row["query"], answer=row["answers"][0], gold=gold,
                        passages=negs[:gold] + [row["positive_passages"][0]] + negs[gold:]))
        if len(out) == n:
            break
    return out


@app.cell
def _():
    pilot_sets = {"k16": nq_pilot_examples(96, 16), "k64": nq_pilot_examples(64, 64)}
    pilot_button = mo.ui.run_button(
        label=f"Run pilot (K=16: {len(pilot_sets['k16'])} questions, K=64: {len(pilot_sets['k64'])} questions; x 2 passes)")
    pilot_button
    return pilot_button, pilot_sets


@app.function
def start_pilot_job(model, tokenizer, pilot_sets, out_dir="/root/models"):
    """Run the pilot in a daemon thread so that a cancelled HTTP request (which interrupts the kernel's
    main thread) cannot stop it. Results go to `<out_dir>/pilot_nq_<regime>.pt` after every question;
    the returned dict shows progress, status and any error."""
    import threading
    import traceback
    state = {"status": "running", "progress": "", "error": None, "t0": time.time()}
    for regime in pilot_sets:
        path = os.path.join(out_dir, f"pilot_nq_{regime}.pt")
        if os.path.exists(path):
            os.remove(path)

    def work():
        try:
            for regime, examples in pilot_sets.items():
                recs = []
                for i, ex in enumerate(examples):
                    rec = dict(gold=ex["gold"])
                    for name, q in (("q", ex["question"]), ("null", "N/A")):
                        rec[name] = extract_teacher_signals(
                            model, build_rag_example(tokenizer, q, ex["passages"], ex["answer"]))
                    recs.append(rec)
                    torch.save(recs, os.path.join(out_dir, f"pilot_nq_{regime}.pt"))
                    state["progress"] = f"{regime} {i + 1}/{len(examples)} ({time.time() - state['t0']:.0f}s)"
            state["status"] = "finished"
        except Exception:
            state["status"] = "error"
            state["error"] = traceback.format_exc()

    state["thread"] = threading.Thread(target=work, name="qsa-pilot", daemon=True)
    state["thread"].start()
    return state


@app.cell
def _(model, pilot_button, pilot_sets, tokenizer):
    mo.stop(not pilot_button.value, mo.md("Press **Run pilot**. It runs in a background thread; results are saved to `/root/models/pilot_nq_<regime>.pt` after every question."))
    use_fast_indexer(True)
    pilot_job = start_pilot_job(model, tokenizer, pilot_sets)
    mo.md("Pilot started in a background thread. Run the next cell to load whatever has finished.")
    return (pilot_job,)


@app.cell
def _(pilot_job, pilot_sets):
    pilot_job["progress"]
    pilot_runs = {_k: torch.load(f"/root/models/pilot_nq_{_k}.pt") for _k in pilot_sets if os.path.exists(f"/root/models/pilot_nq_{_k}.pt")}
    pilot_records = pilot_runs.get("k16")
    mo.md(f"Pilot job: **{pilot_job['status']}**, {pilot_job['progress']}. Loaded: " + ", ".join(f"{_k} = {len(_v)} questions" for _k, _v in pilot_runs.items()))
    return (pilot_runs,)


@app.function
def rank_metrics(scores, gold):
    """scores [N, K]; gold [N] -> MRR, R@1, nDCG@10 (one relevant chunk; ties count against the gold)."""
    g = scores[torch.arange(scores.shape[0]), gold]
    rank = (scores > g[:, None]).sum(-1) + (scores == g[:, None]).sum(-1)
    rr = 1.0 / rank.float()
    ndcg = torch.where(rank <= 10, 1.0 / torch.log2(rank.float() + 1.0), torch.zeros_like(rr))
    return dict(mrr=rr.mean().item(), r_at_1=(rank == 1).float().mean().item(), ndcg_at_10=ndcg.mean().item())


@app.function
def pilot_score_table(records, n_top_heads=16):
    """Gold-rank metrics for every teacher variant in the pilot records.

    Scorers: per-layer indexer (row group x variant x raw/null-calibrated), layer-mean of per-example
    z-scored indexer masses, native top-512 selection share, single dense/sparse heads, and the sum of
    the top `n_top_heads` heads chosen on the other half of the questions (2-fold).
    """
    gold = torch.tensor([r["gold"] for r in records])
    N = len(records)
    stack = lambda part, key: torch.stack([r[part][key] for r in records])
    idx_q, idx_n = stack("q", "idx"), stack("null", "idx")
    sel_q = stack("q", "sel")
    dense_q, dense_n = stack("q", "dense"), stack("null", "dense")
    sparse_q, sparse_n = stack("q", "sparse"), stack("null", "sparse")
    gate_q = stack("q", "gate")
    groups = ["question", "answer", "both"]
    variants = ["mass", "mass_rope", "mass_nope", "mass_paper", "mass_rope_paper", "mass_nope_paper", "mean_score", "max_score"]
    rows = []

    def add(family, layer, head, group, variant, calib, scores):
        rows.append(dict(family=family, layer=layer, head=head, rows=group, variant=variant,
                         calibrated=calib, **rank_metrics(scores, gold)))

    L = idx_q.shape[1]
    for li in range(L):
        for gi, g in enumerate(groups):
            for vi, v in enumerate(variants):
                add("indexer", li, -1, g, v, False, idx_q[:, li, gi, vi])
                add("indexer", li, -1, g, v, True, idx_q[:, li, gi, vi] - idx_n[:, li, gi, vi])
            add("native_selection", li, -1, g, "top512_share", False, sel_q[:, li, gi])

    def z(x):
        return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-12)

    for gi, g in enumerate(groups):
        for calib in (False, True):
            for vi in (0, 3):
                m = idx_q[:, :, gi, vi] - (idx_n[:, :, gi, vi] if calib else 0)
                add("indexer_layer_mean", -1, -1, g, variants[vi], calib, z(m).mean(1))

    H = dense_q.shape[2]
    for fam, tq, tn in (("dense_head", dense_q, dense_n), ("sparse_head", sparse_q, sparse_n)):
        for gi, g in enumerate(groups):
            for calib in (False, True):
                s = tq[:, :, :, gi] - (tn[:, :, :, gi] if calib else 0)
                for li in range(L):
                    for h in range(H):
                        add(fam, li, h, g, "mass", calib, s[:, li, h])
                # cross-validated top heads (QRHead-style selection on the other half)
                half = torch.arange(N) < N // 2
                folds = [(half, ~half), (~half, half)]
                agg_rr = []
                gated_rr = []
                for tr, te in folds:
                    flat = s[tr].reshape(int(tr.sum()), L * H, -1)
                    mrr_tr = torch.tensor([rank_metrics(flat[:, j], gold[tr])["mrr"] for j in range(L * H)])
                    top = mrr_tr.topk(n_top_heads).indices
                    flat_te = s[te].reshape(int(te.sum()), L * H, -1)
                    agg_rr.append((flat_te[:, top].sum(1), gold[te]))
                    gw = gate_q[te][:, :, :, gi].reshape(int(te.sum()), L * H, 1)
                    gated_rr.append(((flat_te[:, top] * gw[:, top]).sum(1), gold[te]))
                for name, parts in ((f"top{n_top_heads}_cv", agg_rr), (f"top{n_top_heads}_cv_gated", gated_rr)):
                    sc = torch.cat([p[0] for p in parts])
                    gd = torch.cat([p[1] for p in parts])
                    rows.append(dict(family=fam + "_" + name, layer=-1, head=-1, rows=g, variant="mass",
                                     calibrated=calib, **rank_metrics(sc, gd)))
                add(fam + "_all_mean", -1, -1, g, "mass", calib, s.mean((1, 2)))
    K = idx_q.shape[-1]
    rand = sum(1.0 / k for k in range(1, K + 1)) / K
    rows.append(dict(family="random", layer=-1, head=-1, rows="-", variant="-", calibrated=False,
                     mrr=rand, r_at_1=1.0 / K, ndcg_at_10=sum(1 / math.log2(k + 1) for k in range(1, 11)) / K))
    return pl.DataFrame(rows)


@app.function
def position_profile(records, part="q", key="idx", group=0, variant=0):
    """Mean teacher mass of NON-gold chunks by slot, per layer -> long DataFrame (layer, slot, mass)."""
    out = []
    for li in range(records[0][part][key].shape[0]):
        acc = torch.zeros(records[0][part][key].shape[-1])
        cnt = torch.zeros_like(acc)
        for r in records:
            m = r[part][key][li, group, variant]
            mask = torch.ones_like(m, dtype=torch.bool)
            mask[r["gold"]] = False
            acc += torch.where(mask, m, torch.zeros_like(m))
            cnt += mask.float()
        for s, v in enumerate((acc / cnt).tolist()):
            out.append(dict(layer=li, slot=s, mass=v))
    return pl.DataFrame(out)


@app.cell
def _(pilot_runs):
    mo.stop(not pilot_runs, mo.md("No pilot results yet."))
    regime_picker = mo.ui.dropdown(options=list(pilot_runs), value=("k64" if "k64" in pilot_runs else list(pilot_runs)[0]), label="Pilot regime")
    regime_picker
    return (regime_picker,)


@app.cell
def _(model, pilot_runs, regime_picker):
    QSA_LAYERS = [li for li, l in enumerate(model.model.layers) if l.layer_type != "linear_attention"]
    pilot_eval = pilot_runs[regime_picker.value]
    pilot_scores = pilot_score_table(pilot_eval).with_columns(
        pl.when(pl.col("layer") >= 0).then(pl.col("layer").map_elements(lambda i: QSA_LAYERS[i], return_dtype=pl.Int64))
        .otherwise(-1).alias("model_layer"))
    _best = lambda fam: (pilot_scores.filter(pl.col("family") == fam).sort("mrr", descending=True).head(1))
    pilot_summary = pl.concat([
        pilot_scores.filter(pl.col("family") == "random"),
        _best("indexer"),
        pilot_scores.filter((pl.col("family") == "indexer_layer_mean")).sort("mrr", descending=True),
        _best("native_selection"),
        _best("dense_head"), _best("sparse_head"),
        pilot_scores.filter(pl.col("family").str.contains("_cv|_all_mean")).sort("mrr", descending=True),
    ])
    mo.vstack([mo.md(f"**Pilot {regime_picker.value}:** {len(pilot_eval)} NQ-dev questions, K = {pilot_eval[0]['q']['idx'].shape[-1]} chunks, gold slot balanced. Ties count against the gold. Top-head sets are chosen on one half of the questions and scored on the other half (2-fold)."),
               mo.ui.table(pilot_summary.drop("head"), selection=None)])
    return QSA_LAYERS, pilot_eval, pilot_scores


@app.cell
def _(pilot_scores):
    _d = (pilot_scores.filter((pl.col("family") == "indexer") & (pl.col("variant") == "mass") & (pl.col("rows") != "both"))
          .with_columns((pl.col("rows") + pl.when(pl.col("calibrated")).then(pl.lit(", null-calibrated")).otherwise(pl.lit(", raw"))).alias("teacher")))
    _ref = pl.concat([
        pilot_scores.filter(pl.col("family") == "random").select(pl.lit("random order").alias("baseline"), "mrr"),
        pilot_scores.filter(pl.col("family") == "dense_head_top16_cv").sort("mrr", descending=True).head(1)
            .select(pl.lit("top-16 dense heads (CV)").alias("baseline"), "mrr"),
    ])
    _colors = alt.Scale(domain=["question, raw", "question, null-calibrated", "answer, raw", "answer, null-calibrated"],
                        range=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])
    _lines = alt.Chart(_d).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
        x=alt.X("model_layer:O", title="QSA layer"),
        y=alt.Y("mrr:Q", title="MRR of the gold chunk", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("teacher:N", scale=_colors, title="Indexer rows"),
        tooltip=["model_layer", "teacher", alt.Tooltip("mrr:Q", format=".3f"), alt.Tooltip("r_at_1:Q", format=".3f"), alt.Tooltip("ndcg_at_10:Q", format=".3f")],
    )
    _rules = alt.Chart(_ref).mark_rule(strokeDash=[4, 4], color="#8a8985").encode(y="mrr:Q", tooltip=["baseline", alt.Tooltip("mrr:Q", format=".3f")])
    _labels = alt.Chart(_ref).mark_text(align="left", dx=4, dy=-6, color="#52514e").encode(y="mrr:Q", x=alt.value(0), text="baseline")
    _end = alt.Chart(_d.filter(pl.col("model_layer") == pl.col("model_layer").max())).mark_text(
        align="left", dx=8, color="#52514e").encode(x="model_layer:O", y="mrr:Q", text="teacher")
    indexer_layer_chart = (_rules + _labels + _lines + _end).properties(
        width=560, height=300, title="Indexer chunk mass as a ranker, by layer (table above has every value)")
    indexer_layer_chart
    return


@app.cell
def _(pilot_scores):
    _h = pilot_scores.filter((pl.col("family") == "dense_head") & (pl.col("rows") == "both") & (~pl.col("calibrated")))
    head_heatmap = alt.Chart(_h).mark_rect(stroke="#fcfcfb", strokeWidth=1).encode(
        x=alt.X("head:O", title="Head"), y=alt.Y("model_layer:O", title="QSA layer"),
        color=alt.Color("mrr:Q", scale=alt.Scale(range=["#cde2fb", "#104281"]), title="MRR"),
        tooltip=["model_layer", "head", alt.Tooltip("mrr:Q", format=".3f"), alt.Tooltip("r_at_1:Q", format=".3f")],
    ).properties(width=560, height=260, title="Single dense head as a ranker (question + answer rows, raw mass)")
    head_heatmap
    return


@app.cell
def _(QSA_LAYERS, pilot_eval):
    _K = pilot_eval[0]['q']['idx'].shape[-1]
    _p = position_profile(pilot_eval).with_columns(
        pl.col("layer").map_elements(lambda i: QSA_LAYERS[i], return_dtype=pl.Int64).alias("model_layer"),
        (pl.col("mass") * _K).alias("rel_mass"))
    position_heatmap = alt.Chart(_p).mark_rect(stroke="#fcfcfb", strokeWidth=1).encode(
        x=alt.X("slot:O", title=f"Chunk slot (0 = first passage, {_K - 1} = next to the question)"),
        y=alt.Y("model_layer:O", title="QSA layer"),
        color=alt.Color("rel_mass:Q", scale=alt.Scale(range=["#cde2fb", "#104281"]), title="Mass x K"),
        tooltip=["model_layer", "slot", alt.Tooltip("rel_mass:Q", format=".2f")],
    ).properties(width=560, height=260, title="Indexer mass on non-gold chunks by slot (question rows, raw); 1.0 = uniform")
    position_heatmap
    return


@app.function
def pilot_cv_report(records, n_boot=4000, seed=0):
    """Fair comparison of teachers on question rows, with every choice made on the other half of the
    questions (2-fold): best indexer layer (raw / null-calibrated), layer mean, native top-512 share,
    and top-16 dense / sparse heads. Adds a paired bootstrap of (indexer CV calibrated - dense top-16)."""
    gold = torch.tensor([r["gold"] for r in records])
    N = len(records)
    S = lambda part, key: torch.stack([r[part][key] for r in records])
    idx_q, idx_n, sel_q = S("q", "idx"), S("null", "idx"), S("q", "sel")
    dense_q, sparse_q = S("q", "dense"), S("q", "sparse")

    def ranks(scores):
        g = scores[torch.arange(N), gold]
        return (scores > g[:, None]).sum(-1) + (scores == g[:, None]).sum(-1)

    half = torch.arange(N) < N // 2
    folds = ((half, ~half), (~half, half))

    def cv_pick(cands):
        out = torch.zeros(N, cands.shape[-1])
        for tr, te in folds:
            mrr = torch.stack([(1.0 / ranks(cands[:, j])[tr].float()).mean() for j in range(cands.shape[1])])
            out[te] = cands[te][:, mrr.argmax()]
        return out

    def cv_top_heads(t, n=16):
        flat = t.reshape(N, -1, t.shape[-1])
        out = torch.zeros(N, t.shape[-1])
        for tr, te in folds:
            mrr = torch.stack([(1.0 / ranks(flat[:, j])[tr].float()).mean() for j in range(flat.shape[1])])
            out[te] = flat[te][:, mrr.topk(n).indices].sum(1)
        return out

    z = lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-12)
    calib = idx_q[:, :, 0, 0] - idx_n[:, :, 0, 0]
    teachers = {
        "indexer, best layer (CV), null-calibrated": cv_pick(calib),
        "indexer, best layer (CV), raw": cv_pick(idx_q[:, :, 0, 0]),
        "indexer, best layer (CV), paper-scale softmax, raw": cv_pick(idx_q[:, :, 0, 3]),
        "indexer, z-scored layer mean, null-calibrated": z(calib).mean(1),
        "native top-512 share, best layer (CV)": cv_pick(sel_q[:, :, 0]),
        "dense heads, top 16 (CV)": cv_top_heads(dense_q[..., 0, :]),
        "sparse heads, top 16 (CV)": cv_top_heads(sparse_q[..., 0, :]),
    }
    rows, rr, nd = [], {}, {}
    for name, t in teachers.items():
        rk = ranks(t).float()
        rr[name] = 1.0 / rk
        nd[name] = torch.where(rk <= 10, 1.0 / torch.log2(rk + 1.0), torch.zeros_like(rk))
        rows.append(dict(teacher=name, mrr=rr[name].mean().item(), ndcg_at_10=nd[name].mean().item(),
                         r_at_1=(rk == 1).float().mean().item()))
    K = idx_q.shape[-1]
    rows.append(dict(teacher="random order", mrr=sum(1.0 / k for k in range(1, K + 1)) / K,
                     ndcg_at_10=sum(1 / math.log2(k + 1) for k in range(1, 11)) / K, r_at_1=1.0 / K))
    gen = torch.Generator().manual_seed(seed)
    b = torch.randint(0, N, (n_boot, N), generator=gen)
    a, c = "indexer, best layer (CV), null-calibrated", "dense heads, top 16 (CV)"
    boot = {}
    for metric, d in (("mrr", rr), ("ndcg_at_10", nd)):
        diff = d[a] - d[c]
        m = diff[b].mean(1)
        boot[metric] = (diff.mean().item(), m.quantile(0.025).item(), m.quantile(0.975).item())
    return pl.DataFrame(rows), boot


@app.cell
def _(pilot_runs):
    _tbls = []
    for _regime, _recs in pilot_runs.items():
        _t, _b = pilot_cv_report(_recs)
        _tbls.append(mo.vstack([
            mo.md(f"**{_regime}** ({len(_recs)} questions). Indexer (CV, calibrated) minus dense top-16 (CV): "
                  f"MRR {_b['mrr'][0]:+.3f} [{_b['mrr'][1]:+.3f}, {_b['mrr'][2]:+.3f}], "
                  f"nDCG@10 {_b['ndcg_at_10'][0]:+.3f} [{_b['ndcg_at_10'][1]:+.3f}, {_b['ndcg_at_10'][2]:+.3f}] (95% paired bootstrap)"),
            mo.ui.table(_t.with_columns(pl.col("mrr", "ndcg_at_10", "r_at_1").round(3)), selection=None, page_size=10),
        ]))
    mo.vstack([mo.md("### Fair comparison (question rows only; every choice made on the other half)")] + _tbls)
    return


@app.cell
def _():
    mo.md(r"""
    ### Pilot findings (4 Oct 2026)

    - **Question rows are the clean test.** In NQ, the gold passage contains the answer string and the BM25
      negatives are filtered to exclude it, so teacher-forced answer rows can find the gold by string match
      (answer-row MRR is about 0.95 at K = 16). The null question keeps the same answer, and null calibration
      drops the answer-row score, which fits that reading.
    - **K = 64 (10.4K tokens, about 20% of blocks kept), the regime that matters:** the indexer with the layer chosen by
      cross-validation and null calibration matches the cross-validated top-16 dense heads of the same model
      (nDCG@10 +1.7 points, 95% CI -1.7 to +5.2, 64 questions). At K = 16 (about 75% of blocks kept) it trails
      them (-3.6 points, CI -6.4 to -1.0, 96 questions).
    - **Depth profile:** the signal is near random at layer 3 and peaks at layers 31 and 35, then falls toward layer 47.
    - **Null calibration is required:** it adds 0.1 to 0.2 MRR at mid-depth layers for K = 64.
    - **Position bias goes toward the first passages, not the recent ones:** the raw mass on non-gold chunks falls
      with the slot index (slope x (K - 1) / gold margin = -0.27 at layer 31, K = 64). Null calibration
      removes most of it (-0.07).
    - **The RoPE / position-free split does not give a free de-biased teacher:** the position-free half alone
      ranks worse than the full score at mid and late layers, and the RoPE half alone is weak.
    - **Native top-512 selection is a weak ranker** (best-layer MRR 0.42 at K = 64) even though gold blocks
      are selected at 2.7x the base rate.
    - **Output-gate weighting changes nothing** at the chunk level (same ranks with and without it).
    """)
    return


@app.function
def profile_forward(model, ids):
    """Wall time per module family for one forward pass (CUDA-synchronized pre/post hooks)."""
    times, starts, hooks = {}, {}, []
    fams = {"experts": NVFP4Experts, "gdn": qm.Qwen4ExpTextGatedDeltaNet, "qsa_attention": qm.Qwen4ExpTextAttention,
            "qsa_indexer": qm.Qwen4ExpTextQSAIndexer, "ple": qm.Qwen4ExpTextPLELayer}
    for name, mod in model.named_modules():
        for fam, cls in fams.items():
            if isinstance(mod, cls):
                def pre(m, a, fam=fam, name=name):
                    torch.cuda.synchronize(); starts[name] = time.time()
                def post(m, a, o, fam=fam, name=name):
                    torch.cuda.synchronize(); times[fam] = times.get(fam, 0.0) + time.time() - starts[name]
                hooks += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
    torch.cuda.synchronize(); t0 = time.time()
    try:
        with torch.no_grad():
            model.model(ids)
    finally:
        for h in hooks:
            h.remove()
    torch.cuda.synchronize()
    total = time.time() - t0
    times["other"] = total - sum(v for k, v in times.items() if k != "qsa_indexer")
    return dict(total_s=total, **{k + "_s": v for k, v in times.items()})


@app.cell
def _():
    mo.md(r"""
    ## Speed path

    Profiling showed the NVFP4 expert path at 67% to 89% of each forward pass. The PyTorch decode makes many passes with int64
    temporaries, and the per-expert loop launches about 200K small kernels per pass. The speed path keeps the same W4A16 math:

    - `nvfp4_dequant_kernel` (Triton) decodes NVFP4 to bf16 at memory bandwidth into a reusable buffer.
    - `grouped_experts_forward` sorts the (token, expert) pairs and runs all experts of a chunk with `torch._grouped_mm`.
    - `use_fla_gdn` swaps the Gated DeltaNet reference loops for the `flash-linear-attention` Triton kernels.

    Each patch is a switch, so the original path stays available for parity checks.
    """)
    return


@app.cell
def _():
    import triton
    import triton.language as tl

    return tl, triton


@app.cell
def _(tl, triton):
    @triton.jit
    def nvfp4_e2m1_to_f32(c):
        mag = c & 7
        e = mag >> 1
        m = (mag & 1).to(tl.float32)
        p = tl.where(e == 1, 1.0, tl.where(e == 2, 2.0, 4.0))
        v = tl.where(e == 0, 0.5 * m, (1.0 + 0.5 * m) * p)
        return tl.where(c >= 8, -v, v)

    return (nvfp4_e2m1_to_f32,)


@app.cell
def _(nvfp4_e2m1_to_f32, tl, triton):
    @triton.jit
    def nvfp4_dequant_kernel(q_ptr, s_ptr, g_ptr, out_ptr, KH, KS, ROWS_PER_G, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cb = tl.program_id(1)
        j = cb * BLOCK + tl.arange(0, BLOCK)
        mask = j < KH
        b = tl.load(q_ptr + row * KH + j, mask=mask, other=0).to(tl.int32)
        s8 = tl.load(s_ptr + row * KS + j // 8, mask=mask, other=0)
        sc = s8.to(tl.float8e4nv, bitcast=True).to(tl.float32) * tl.load(g_ptr + row // ROWS_PER_G)
        lo = nvfp4_e2m1_to_f32(b & 15) * sc
        hi = nvfp4_e2m1_to_f32(b >> 4) * sc
        out = tl.reshape(tl.join(lo, hi), (2 * BLOCK,))
        o = 2 * cb * BLOCK + tl.arange(0, 2 * BLOCK)
        tl.store(out_ptr + row * (2 * KH) + o, out.to(tl.bfloat16), mask=o < 2 * KH)

    return (nvfp4_dequant_kernel,)


@app.cell
def _(nvfp4_dequant_kernel, triton):
    def dequant_nvfp4_triton(w_u8, scale_u8, scale2_groups, rows_per_group, out=None, block=256):
        """w_u8 [G?, N, K/2] uint8, scale_u8 [.., N, K/16], scale2_groups fp32 flat [rows / rows_per_group].
        Same math and rounding as dequant_nvfp4: bf16(e2m1 * (fp8(scale) * scale2))."""
        N2 = w_u8.shape[-1]
        rows = w_u8.numel() // N2
        if out is None:
            out = torch.empty(*w_u8.shape[:-1], 2 * N2, dtype=torch.bfloat16, device=w_u8.device)
        grid = (rows, triton.cdiv(N2, block))
        nvfp4_dequant_kernel[grid](w_u8, scale_u8, scale2_groups, out, N2, scale_u8.shape[-1], rows_per_group, BLOCK=block)
        return out

    return (dequant_nvfp4_triton,)


@app.cell
def _(dequant_nvfp4_triton):
    def grouped_experts_forward(self, hidden_states, top_k_index, top_k_weights):
        """NVFP4Experts forward: Triton dequant of the experts that received tokens (into a reusable
        buffer), torch._grouped_mm per chunk of experts, then one weighted bmm to combine the K outputs
        of each token (W4A16, same math as the per-expert loop)."""
        T, K = top_k_index.shape
        E, H, I = self.num_experts, self.hidden_dim, self.intermediate_dim
        chunk = getattr(self, "grouped_chunk", 256)
        dev = hidden_states.device
        flat = top_k_index.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=E)
        tok = order // K
        xs = hidden_states[tok]
        hit = torch.nonzero(counts).flatten()
        n_hit = hit.numel()
        all_hit = n_hit == E
        ends = counts[hit].cumsum(0)
        ends_l = ends.tolist()
        cache = type(self).__dict__.get("grouped_buffers")
        if cache is None:
            cache = {}
            type(self).grouped_buffers = cache
        buf = cache.get((dev, chunk))
        if buf is None:
            buf = (torch.empty(chunk, 2 * I, H, dtype=torch.bfloat16, device=dev),
                   torch.empty(chunk, H, I, dtype=torch.bfloat16, device=dev))
            cache[(dev, chunk)] = buf
        out_sorted = torch.empty(T * K, H, dtype=hidden_states.dtype, device=dev)
        for c0 in range(0, n_hit, chunk):
            c1 = min(n_hit, c0 + chunk)
            n = c1 - c0
            r0 = ends_l[c0 - 1] if c0 > 0 else 0
            r1 = ends_l[c1 - 1]
            if all_hit:
                sl = slice(c0, c1)
                gq, gs, gg = self.gu_q[sl], self.gu_s[sl], self.gu_g[sl]
                dq, ds, dg = self.d_q[sl], self.d_s[sl], self.d_g[sl]
            else:
                ids = hit[c0:c1]
                gq, gs, gg = self.gu_q.index_select(0, ids), self.gu_s.index_select(0, ids), self.gu_g.index_select(0, ids)
                dq, ds, dg = self.d_q.index_select(0, ids), self.d_s.index_select(0, ids), self.d_g.index_select(0, ids)
            wgu = dequant_nvfp4_triton(gq, gs, gg.reshape(-1), I, out=buf[0][:n])
            wd = dequant_nvfp4_triton(dq, ds, dg, H, out=buf[1][:n])
            offs = (ends[c0:c1] - r0).to(torch.int32)
            h = torch._grouped_mm(xs[r0:r1], wgu.transpose(1, 2), offs=offs)
            h = F.silu(h[:, :I]) * h[:, I:]
            out_sorted[r0:r1] = torch._grouped_mm(h, wd.transpose(1, 2), offs=offs)
        # Combine the K expert outputs of each token: back to (token, slot) order, then one weighted bmm
        # (fp32 accumulation, same bf16 result as an fp32 index_add followed by the bf16 cast).
        unsorted = torch.empty_like(out_sorted)
        unsorted[order] = out_sorted
        return torch.bmm(top_k_weights.to(unsorted.dtype)[:, None, :], unsorted.view(T, K, H)).squeeze(1)

    return (grouped_experts_forward,)


@app.cell
def _(grouped_experts_forward):
    def use_grouped_experts(enabled=True, chunk=256):
        """Switch NVFP4Experts.forward between the grouped-GEMM path and the original per-expert loop."""
        cls = NVFP4Experts
        if "loop_forward" not in cls.__dict__:
            cls.loop_forward = cls.forward
        cls.grouped_chunk = chunk
        cls.forward = grouped_experts_forward if enabled else cls.loop_forward
        if not enabled and "grouped_buffers" in cls.__dict__:
            cls.grouped_buffers.clear()
        return enabled

    return (use_grouped_experts,)


@app.function
def use_fla_gdn(enabled=True):
    """Rebind the Gated DeltaNet chunk / recurrent functions of the loaded qwen4_exp module to the
    flash-linear-attention kernels (transformers binds them at import time, before fla was installed)."""
    import inspect
    if "reference_chunk_gdr" not in qm.__dict__:
        qm.reference_chunk_gdr = qm.torch_chunk_gated_delta_rule
        qm.reference_recurrent_gdr = qm.torch_recurrent_gated_delta_rule
    if not enabled:
        qm.torch_chunk_gated_delta_rule = qm.reference_chunk_gdr
        qm.torch_recurrent_gated_delta_rule = qm.reference_recurrent_gdr
        return False
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    def wrap(fn):
        params = set(inspect.signature(fn).parameters)
        def call(*args, **kwargs):
            return fn(*args, **{k: v for k, v in kwargs.items() if k in params})
        return call

    qm.torch_chunk_gated_delta_rule = wrap(chunk_gated_delta_rule)
    qm.torch_recurrent_gated_delta_rule = wrap(fused_recurrent_gated_delta_rule)
    return True


@app.function
def signals_from_store(model, store, ex):
    """Chunk-level teacher signals from captured attention inputs (same outputs as extract_teacher_signals).

    store[layer] = (x [1, T, D], (cos, sin) [1, T, rot]) covering every position of ex["ids"].
    """
    device = model.lm_head.weight.device
    spans = ex["spans"]
    nq, na = ex["q_rows"].numel(), ex["a_rows"].numel()
    rows = torch.cat([ex["q_rows"], ex["a_rows"]]).to(device)
    groups = [slice(0, nq), slice(nq, nq + na), slice(0, nq + na)]
    out = {k: [] for k in ("idx", "sel", "dense", "sparse", "gate", "gnorm")}
    for li in sorted(store):
        x, (cos, sin) = store[li]
        x, cos, sin = x[0], cos[0], sin[0]
        attn = model.model.layers[li].self_attn
        r, d = attn.indexer.compress_ratio, attn.indexer.index_head_dim
        tok, sc, selb = indexer_selection_tokens(attn.indexer, x, cos, sin, rows)
        valid = sc["valid"]
        neg = float("-inf")
        s_rope = (torch.relu(sc["dots_rope"]).sum(-1) / math.sqrt(d)).masked_fill(~valid, neg)
        s_nope = (torch.relu(sc["dots_nope"]).sum(-1) / math.sqrt(d)).masked_fill(~valid, neg)
        full = sc["score"]
        masses = [torch.stack([(v * scale).softmax(-1)[:, s // r : e // r].sum(-1) for s, e in spans], -1)
                  for scale in (1.0, math.sqrt(d)) for v in (full, s_rope, s_nope)]
        finite = full.masked_fill(~valid, 0.0)
        mean_sc = torch.stack([finite[:, s // r : e // r].mean(-1) for s, e in spans], -1)
        max_sc = torch.stack([full[:, s // r : e // r].amax(-1) for s, e in spans], -1)
        idx = torch.stack(masses + [mean_sc, max_sc], 1)
        sel = torch.stack([selb[:, s // r : e // r].float().mean(-1) for s, e in spans], -1)
        sp = attention_side_path(attn, x, cos, sin, rows, tok)
        de = attention_side_path(attn, x, cos, sin, rows, None)
        dense, sparse = chunk_mass(de["probs"], spans), chunk_mass(sp["probs"], spans)
        gate = sp["gate"].mean(-1)
        gnorm = (sp["o"].float() * sp["gate"]).norm(dim=-1)
        out["idx"].append(torch.stack([idx[g].mean(0) for g in groups]))
        out["sel"].append(torch.stack([sel[g].mean(0) for g in groups]))
        out["dense"].append(torch.stack([dense[g].mean(0) for g in groups], 1))
        out["sparse"].append(torch.stack([sparse[g].mean(0) for g in groups], 1))
        out["gate"].append(torch.stack([gate[g].mean(0) for g in groups], 1))
        out["gnorm"].append(torch.stack([gnorm[g].mean(0) for g in groups], 1))
    return {k: torch.stack(v).float().cpu() for k, v in out.items()}


@app.function
@torch.no_grad()
def extract_teacher_pair(model, examples):
    """Teacher signals for several variants of one prompt that share everything before the question
    (e.g. the real question and the null question). The shared prefix runs once with a KV cache;
    each variant then runs only its own suffix on a copy of the cache. Returns one dict per variant."""
    import copy
    device = model.lm_head.weight.device
    q0 = int(examples[0]["q_rows"][0])
    prefix = examples[0]["ids"][:, :q0]
    for ex in examples[1:]:
        if int(ex["q_rows"][0]) != q0 or not torch.equal(ex["ids"][:, :q0], prefix):
            raise ValueError("variants must share the prefix before the question")
    store_p, hooks = capture_attention_inputs(model)
    try:
        cache = model.model(prefix.to(device), use_cache=True).past_key_values
    finally:
        for h in hooks:
            h.remove()
    results = []
    for i, ex in enumerate(examples):
        c = cache if i == len(examples) - 1 else copy.deepcopy(cache)
        store_s, hooks = capture_attention_inputs(model)
        try:
            model.model(ex["ids"][:, q0:].to(device), past_key_values=c, use_cache=True)
        finally:
            for h in hooks:
                h.remove()
        store = {li: (torch.cat([store_p[li][0], store_s[li][0]], dim=1), store_s[li][1]) for li in store_s}
        results.append(signals_from_store(model, store, ex))
        del store, store_s, c
    return results


@app.function
def expand_cache_batch(cache, n):
    """Repeat a batch-1 DynamicCache (attention KV, indexer keys, GDN conv / recurrent states, n-gram
    context, bound position ids) to batch n, in place."""
    def rep(t):
        return t.expand(n, *t.shape[1:]).contiguous() if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == 1 else t
    for layer in cache.layers:
        for name, val in list(vars(layer).items()):
            if torch.is_tensor(val):
                setattr(layer, name, rep(val))
            elif isinstance(val, dict):
                setattr(layer, name, {k: rep(v) for k, v in val.items()})
    pid = getattr(cache, "position_ids", None)
    if pid is not None:
        cache.position_ids = pid.expand(pid.shape[0], n, pid.shape[2]).contiguous()
    return cache


@app.function
@torch.no_grad()
def extract_teacher_batch(model, examples, pad_id=198):
    """Teacher signals for prompt variants that share everything before the question: one prefix pass
    with a KV cache, then all suffixes together as one right-padded batch on the expanded cache.
    Right padding is safe: every row we read lies before its padding and attention is causal."""
    device = model.lm_head.weight.device
    q0 = int(examples[0]["q_rows"][0])
    prefix = examples[0]["ids"][:, :q0]
    for ex in examples[1:]:
        if int(ex["q_rows"][0]) != q0 or not torch.equal(ex["ids"][:, :q0], prefix):
            raise ValueError("variants must share the prefix before the question")
    store_p, hooks = capture_attention_inputs(model)
    try:
        cache = model.model(prefix.to(device), use_cache=True).past_key_values
    finally:
        for h in hooks:
            h.remove()
    suffixes = [ex["ids"][0, q0:] for ex in examples]
    L = max(s.numel() for s in suffixes)
    batch = torch.full((len(examples), L), pad_id, dtype=torch.long)
    for i, s in enumerate(suffixes):
        batch[i, : s.numel()] = s
    expand_cache_batch(cache, len(examples))
    store_s, hooks = capture_attention_inputs(model)
    try:
        model.model(batch.to(device), past_key_values=cache, use_cache=True)
    finally:
        for h in hooks:
            h.remove()
    del cache
    results = []
    for i, ex in enumerate(examples):
        n_i = suffixes[i].numel()
        t_i = q0 + n_i
        store = {li: (torch.cat([store_p[li][0], store_s[li][0][i : i + 1, :n_i]], dim=1),
                      (store_s[li][1][0][i : i + 1, :t_i], store_s[li][1][1][i : i + 1, :t_i]))
                 for li in store_s}
        results.append(signals_from_store(model, store, ex))
        del store
    return results


@app.function
def repeat_cache_rows(cache, v):
    """Repeat every batch row of a DynamicCache v times (row b -> rows b*v .. b*v+v-1), in place."""
    def rep(t):
        return t.repeat_interleave(v, dim=0) if torch.is_tensor(t) and t.dim() > 0 else t
    for layer in cache.layers:
        for name, val in list(vars(layer).items()):
            if torch.is_tensor(val):
                setattr(layer, name, rep(val))
            elif isinstance(val, dict):
                setattr(layer, name, {k: rep(x) for k, x in val.items()})
    pid = getattr(cache, "position_ids", None)
    if pid is not None:
        cache.position_ids = pid.repeat_interleave(v, dim=1)
    return cache


@app.function
@torch.no_grad()
def extract_teacher_multi(model, groups, pad_id=198):
    """Teacher signals for several questions at once. groups[i] is the list of prompt variants of
    question i (e.g. [question, null]); variants of one question share everything before the
    question, and all questions must have the same prefix length. One batched prefix pass with a KV
    cache, then one right-padded batch with every variant's suffix. Returns results[i][variant]."""
    device = model.lm_head.weight.device
    Q, V = len(groups), len(groups[0])
    q0 = int(groups[0][0]["q_rows"][0])
    prefixes = []
    for g in groups:
        if len(g) != V or any(int(ex["q_rows"][0]) != q0 for ex in g):
            raise ValueError("all questions need the same number of variants and the same prefix length")
        if any(not torch.equal(ex["ids"][:, :q0], g[0]["ids"][:, :q0]) for ex in g[1:]):
            raise ValueError("variants of one question must share the prefix")
        prefixes.append(g[0]["ids"][:, :q0])
    store_p, hooks = capture_attention_inputs(model)
    try:
        cache = model.model(torch.cat(prefixes).to(device), use_cache=True).past_key_values
    finally:
        for h in hooks:
            h.remove()
    suffixes = [ex["ids"][0, q0:] for g in groups for ex in g]
    L = max(s.numel() for s in suffixes)
    batch = torch.full((Q * V, L), pad_id, dtype=torch.long)
    for i, s in enumerate(suffixes):
        batch[i, : s.numel()] = s
    repeat_cache_rows(cache, V)
    store_s, hooks = capture_attention_inputs(model)
    try:
        model.model(batch.to(device), past_key_values=cache, use_cache=True)
    finally:
        for h in hooks:
            h.remove()
    del cache
    results = []
    for qi, g in enumerate(groups):
        res_q = []
        for vi, ex in enumerate(g):
            row = qi * V + vi
            n_i = suffixes[row].numel()
            t_i = q0 + n_i
            store = {li: (torch.cat([store_p[li][0][qi : qi + 1], store_s[li][0][row : row + 1, :n_i]], dim=1),
                          (store_s[li][1][0][row : row + 1, :t_i], store_s[li][1][1][row : row + 1, :t_i]))
                     for li in store_s}
            res_q.append(signals_from_store(model, store, ex))
            del store
        results.append(res_q)
    return results


@app.function
def use_compiled_hyper_connections(enabled=True):
    """torch.compile the gated-residual (hyper-connection) forward shared by all 97 instances.
    Module parameters are graph inputs (inline_inbuilt_nn_modules), so instances share one graph."""
    import torch._dynamo
    cls = qm.Qwen4ExpTextGatedResidual
    if "eager_forward" not in cls.__dict__:
        cls.eager_forward = cls.forward
    if not enabled:
        cls.forward = cls.eager_forward
        return False
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    if "compiled_forward" not in cls.__dict__:
        cls.compiled_forward = torch.compile(cls.eager_forward, dynamic=None)
    cls.forward = cls.compiled_forward
    return True


@app.function
def use_flex_attention(enabled=True, min_query_len=1024):
    """Route QSA attention with >= min_query_len query rows through FlexAttention with a block mask
    built from the boolean (causal & indexer) mask; shorter queries keep SDPA."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    reg = ALL_ATTENTION_FUNCTIONS._global_mapping
    if "sdpa_reference" not in qm.__dict__:
        qm.sdpa_reference = reg["sdpa"]
    if not enabled:
        reg["sdpa"] = qm.sdpa_reference
        return False
    if "flex_compiled" not in qm.__dict__:
        qm.flex_compiled = torch.compile(flex_attention, dynamic=False)

    def sdpa_or_flex(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        if (query.shape[2] < min_query_len or attention_mask is None or attention_mask.dtype != torch.bool
                or not isinstance(module, qm.Qwen4ExpTextAttention)):
            return qm.sdpa_reference(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        B, _, Tq, _ = query.shape
        Tk = key.shape[2]
        m = attention_mask[:, 0]

        def mask_mod(b, h, q_idx, kv_idx):
            return m[b, q_idx, kv_idx]

        bm = create_block_mask(mask_mod, B, None, Tq, Tk, device=query.device)
        out = qm.flex_compiled(query, key, value, block_mask=bm, scale=scaling, enable_gqa=True)
        return out.transpose(1, 2).contiguous(), None

    reg["sdpa"] = sdpa_or_flex
    return True


@app.cell
def _(use_grouped_experts):
    def use_fast_path(enabled=True):
        """All speed switches at once: grouped NVFP4 experts, fla GDN kernels, vectorized indexer,
        compiled hyper-connections, FlexAttention for long queries. enabled=False restores the reference path."""
        use_grouped_experts(enabled)
        use_fla_gdn(enabled)
        use_fast_indexer(enabled)
        use_compiled_hyper_connections(enabled)
        use_flex_attention(enabled)
        return enabled

    return (use_fast_path,)


@app.function
def start_pilot_job_fast(model, tokenizer, pilot_sets, out_dir="/root/models", tag="fast", questions_per_batch=2):
    """Pilot in a background thread with the fast extraction: questions are processed in batches of
    `questions_per_batch` (one batched prefix pass + one batched suffix pass per batch, the question
    and the null question share the prefix). Results go to `<out_dir>/pilot_<tag>_nq_<regime>.pt`."""
    import threading
    import traceback
    state = {"status": "running", "progress": "", "error": None, "t0": time.time(), "per_regime_s": {}}
    for regime in pilot_sets:
        path = os.path.join(out_dir, f"pilot_{tag}_nq_{regime}.pt")
        if os.path.exists(path):
            os.remove(path)

    def work():
        try:
            for regime, examples in pilot_sets.items():
                t_reg = time.time()
                recs = []
                for b0 in range(0, len(examples), questions_per_batch):
                    batch = examples[b0 : b0 + questions_per_batch]
                    groups = [[build_rag_example(tokenizer, q, ex["passages"], ex["answer"]) for q in (ex["question"], "N/A")]
                              for ex in batch]
                    for ex, (sig_q, sig_null) in zip(batch, extract_teacher_multi(model, groups)):
                        recs.append(dict(gold=ex["gold"], q=sig_q, null=sig_null))
                    state["progress"] = f"{regime} {len(recs)}/{len(examples)} ({time.time() - state['t0']:.0f}s)"
                torch.save(recs, os.path.join(out_dir, f"pilot_{tag}_nq_{regime}.pt"))
                state["per_regime_s"][regime] = time.time() - t_reg
            state["status"] = "finished"
        except Exception:
            state["status"] = "error"
            state["error"] = traceback.format_exc()

    state["thread"] = threading.Thread(target=work, name=f"qsa-pilot-{tag}", daemon=True)
    state["thread"].start()
    return state


@app.cell
def _():
    fast_pilot_button = mo.ui.run_button(label="Run fast pilot (both regimes, 3 questions per pass)")
    fast_pilot_button
    return (fast_pilot_button,)


@app.cell
def _(fast_pilot_button, model, pilot_sets, tokenizer, use_fast_path):
    mo.stop(not fast_pilot_button.value, mo.md("Press **Run fast pilot**. It runs in a background thread; results go to `/root/models/pilot_fast_nq_<regime>.pt`."))
    use_fast_path(True)
    fast_pilot_job = start_pilot_job_fast(model, tokenizer, pilot_sets, questions_per_batch=3)
    mo.md("Fast pilot started in a background thread. Run the next cell to compare with the reference-path pilot.")
    return (fast_pilot_job,)


@app.cell
def _(fast_pilot_job, pilot_runs, pilot_sets):
    fast_pilot_job["progress"]
    fast_runs = {_k: torch.load(f"/root/models/pilot_fast_nq_{_k}.pt") for _k in pilot_sets if os.path.exists(f"/root/models/pilot_fast_nq_{_k}.pt")}
    _rows = []
    for _k, _recs in fast_runs.items():
        _tf, _bf = pilot_cv_report(_recs)
        _ts, _bs = pilot_cv_report(pilot_runs[_k])
        _j = _ts.join(_tf, on="teacher", suffix="_fast")
        _rows.append(mo.vstack([
            mo.md(f"**{_k}**: reference path {_k} vs fast path ({fast_pilot_job['per_regime_s'].get(_k, float('nan')):.0f} s for {len(_recs)} questions). "
                  f"Indexer (CV, calibrated) minus dense top-16: reference nDCG {_bs['ndcg_at_10'][0]:+.3f}, fast nDCG {_bf['ndcg_at_10'][0]:+.3f}."),
            mo.ui.table(_j.select("teacher", "mrr", "mrr_fast", "ndcg_at_10", "ndcg_at_10_fast").with_columns(pl.col("mrr", "mrr_fast", "ndcg_at_10", "ndcg_at_10_fast").round(3)), selection=None, page_size=10),
        ]))
    mo.vstack([mo.md(f"Fast pilot: **{fast_pilot_job['status']}**, {fast_pilot_job['progress']}")] + _rows)
    return


@app.cell
def _():
    mo.md(r"""
    ### Speed path results (4 Oct 2026)

    | Measure | Reference path | Fast path |
    |---|---:|---:|
    | One 10,260-token forward pass (one sequence) | 7.0 s | 2.28 s |
    | One K = 64 question, question + null pass | 27.3 s | 2.27 s (3 questions per batch) |
    | Pilot K = 16, 96 questions | 864 s | 94 s |
    | Pilot K = 64, 64 questions | 1,750 s | 145 s |

    Steps, each a switch inside `use_fast_path` (forward at 10,260 tokens): grouped NVFP4 experts with the Triton
    decode (7.0 s to 3.6 s, bit-identical logits), fla GDN kernels (to 2.79 s), compiled hyper-connections
    (to 2.47 s), FlexAttention for long queries (to 2.28 s). On top of that, the pilot shares one prefix pass between the
    question and the null question, runs both suffixes as one right-padded batch, and processes 3 questions per pass.

    **Parity.** The grouped experts reproduce the per-expert loop exactly (logit KL 0). The fla GDN kernels, the
    compiled hyper-connections, FlexAttention and the KV-cache continuation each change the bf16 numeric path (each step
    shifts WikiText logits by a KL of about 0.015 to 0.02; top-1 agreement about 96%). On the teacher signal, the fast pilot
    agrees with the reference pilot: Spearman of the indexer chunk masses 0.993 (K = 16) and 0.996 (K = 64), the same gold rank for 81% to 91% of
    questions at layers 31 and 35, and cross-validated MRR / nDCG within 0.02 for every teacher except the native
    top-512 share at K = 64 (0.42 to 0.34; that teacher counts discrete block shares and flips on small score changes).
    """)
    return


if __name__ == "__main__":
    app.run()
