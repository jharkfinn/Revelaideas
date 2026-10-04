# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "flash-linear-attention==0.5.2",
#     "numpy==2.4.6",
#     "pytrec-eval-terrier==0.5.10",
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
    if Tk // r <= self.block_topk:
        # Every complete block is selected and the tail tokens are always kept, so the selection mask equals
        # the causal mask. Only the cache needs the raw indexer keys.
        if past_key_values is not None:
            token_k = torch.split(self.index_qk_proj(hidden_states), [Hq * d, d], dim=-1)[1]
            past_key_values.update_indexer(token_k.reshape(B, Tq, d), self.layer_idx)
        if attention_mask.is_floating_point():
            return torch.where(vis, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
        return vis
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
            silu_mul_fn = type(self).__dict__.get("silu_mul_fn")
            h = silu_mul_fn(h, I) if silu_mul_fn is not None else F.silu(h[:, :I]) * h[:, I:]
            out_sorted[r0:r1] = torch._grouped_mm(h, wd.transpose(1, 2), offs=offs)
        combine_fn = type(self).__dict__.get("combine_fn")
        if combine_fn is not None:
            # compiled gather + weighted sum (use_fused_glue)
            inv = torch.empty_like(order)
            inv[order] = torch.arange(order.numel(), device=dev)
            return combine_fn(out_sorted, inv, top_k_weights, T, K)
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


@app.cell(hide_code=True)
def _():
    class StopForward(Exception):
        """Raised by a forward hook to end a forward pass after the last decoder layer that is needed."""


    def run_until(model, ids, last_layer=None, **kwargs):
        """model.model(ids, **kwargs), stopped after decoder layer `last_layer` (None = all layers).

        Returns the KV cache when use_cache=True (pass past_key_values to continue a cache). Layers after
        last_layer are never computed, and their cache entries stay empty, so a continuation must use the
        same last_layer."""
        n = len(model.model.layers)
        if last_layer is None or last_layer >= n - 1:
            return model.model(ids, **kwargs).past_key_values
        cache = kwargs.pop("past_key_values", None)
        if cache is None and kwargs.get("use_cache"):
            cache = qm.DynamicCache(config=model.model.config)

        def stop(module, args, output):
            raise StopForward

        hook = model.model.layers[last_layer].register_forward_hook(stop)
        try:
            model.model(ids, past_key_values=cache, **kwargs)
        except StopForward:
            pass
        finally:
            hook.remove()
        return cache


    return (run_until,)


@app.cell
def extract_teacher_multi(run_until):
    @torch.no_grad()
    def extract_teacher_multi(model, groups, pad_id=198, signals_fn=None, last_layer=None):
        """Teacher signals for several questions at once. groups[i] is the list of prompt variants of
        question i (e.g. [question, null]); variants of one question share everything before the
        question, and all questions must have the same prefix length. One batched prefix pass with a KV
        cache, then one right-padded batch with every variant's suffix. Returns results[i][variant].
        signals_fn(model, store, ex) computes the signals of one variant (default: signals_from_store).
        last_layer: stop both passes after this decoder layer (None = all 48 layers); only the QSA layers up to it
        are captured."""
        signals_fn = signals_fn or signals_from_store
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
            cache = run_until(model, torch.cat(prefixes).to(device), last_layer, use_cache=True)
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
            run_until(model, batch.to(device), last_layer, past_key_values=cache, use_cache=True)
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
                res_q.append(signals_fn(model, store, ex))
                del store
            results.append(res_q)
        return results

    return (extract_teacher_multi,)


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


@app.function(hide_code=True)
def bool_mask_to_block_mask(m, mask_mod, block=128):
    """FlexAttention BlockMask from a dense boolean mask m [B, Tq, Tk].

    Same result as create_block_mask(mask_mod, ...), without its vmap over every (q, kv) pair, which
    builds int64 index grids of B * Tq * Tk elements (about 5 GiB at 25K tokens). Padded tail positions
    count as masked, so tail blocks are partial and the kernel applies mask_mod inside them."""
    from torch.nn.attention.flex_attention import BlockMask
    B, Tq, Tk = m.shape
    nq, nk = -(-Tq // block), -(-Tk // block)
    mp = torch.zeros(B, nq * block, nk * block, dtype=torch.bool, device=m.device)
    mp[:, :Tq, :Tk] = m
    blk = mp.view(B, nq, block, nk, block)
    any_b = blk.any(4).any(2)
    all_b = blk.all(4).all(2)
    del mp, blk

    def to_idx(x):
        num = x.sum(-1, dtype=torch.int32)[:, None].contiguous()
        idx = torch.argsort(x.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)[:, None].contiguous()
        return num, idx

    kv_num, kv_idx = to_idx(any_b & ~all_b)
    full_num, full_idx = to_idx(all_b)
    return BlockMask.from_kv_blocks(kv_num, kv_idx, full_num, full_idx, BLOCK_SIZE=block, mask_mod=mask_mod,
                                    seq_lengths=(Tq, Tk))


@app.function
def use_flex_attention(enabled=True, min_query_len=1024):
    """Route QSA attention with >= min_query_len query rows through FlexAttention with a block mask
    built from the boolean (causal & indexer) mask; shorter queries keep SDPA. The compiled kernel
    accepts variable sequence lengths (one static compile, then one dynamic compile)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from torch.nn.attention.flex_attention import flex_attention
    reg = ALL_ATTENTION_FUNCTIONS._global_mapping
    if "sdpa_reference" not in qm.__dict__:
        qm.sdpa_reference = reg["sdpa"]
    if not enabled:
        reg["sdpa"] = qm.sdpa_reference
        return False
    if "flex_compiled_auto" not in qm.__dict__:
        qm.flex_compiled_auto = torch.compile(flex_attention, dynamic=None)

    def sdpa_or_flex(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        if (query.shape[2] < min_query_len or attention_mask is None or attention_mask.dtype != torch.bool
                or not isinstance(module, qm.Qwen4ExpTextAttention)):
            return qm.sdpa_reference(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        m = attention_mask[:, 0]

        def mask_mod(b, h, q_idx, kv_idx):
            return m[b, q_idx, kv_idx]

        bm = bool_mask_to_block_mask(m, mask_mod)
        out = qm.flex_compiled_auto(query, key, value, block_mask=bm, scale=scaling, enable_gqa=True)
        return out.transpose(1, 2).contiguous(), None

    reg["sdpa"] = sdpa_or_flex
    return True


@app.cell(hide_code=True)
def _(tl, triton):
    @triton.jit
    def qsa_attn_step(q, acc, m_i, l_i, kbase, vbase, tok, ok, sk_t, sv_t, d, sm_scale):
        """One online-softmax step of qsa_sparse_attn_kernel over the key slots tok (ok = slot is used)."""
        tok64 = tok.to(tl.int64)
        k = tl.load(kbase + tok64[:, None] * sk_t + d[None, :], mask=ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vbase + tok64[:, None] * sv_t + d[None, :], mask=ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        return acc, m_new, l_i


    return (qsa_attn_step,)


@app.cell(hide_code=True)
def _(qsa_attn_step, tl, triton):
    @triton.jit
    def qsa_sparse_attn_kernel(q_ptr, k_ptr, v_ptr, idx_ptr, o_ptr, sm_scale, q_off,
                               sq_b, sq_h, sq_t, sk_b, sk_h, sk_t, sv_b, sv_h, sv_t, si_b, si_t, so_b, so_t, so_h,
                               TOPK: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr,
                               R: tl.constexpr, NB: tl.constexpr):
        """QSA sparse attention for one query row and one KV head: the G query heads of that KV head attend
        to the row's selected blocks (idx, block ids, -1 = none, R tokens per block) and to its tail tokens
        ((pos + 1) // R * R .. pos). Same key set as the indexer mask of Qwen4ExpTextQSAIndexer."""
        t = tl.program_id(0)
        hk = tl.program_id(1)
        b = tl.program_id(2).to(tl.int64)
        pos = q_off + t
        hs = tl.arange(0, GP)
        hmask = hs < G
        d = tl.arange(0, D)
        q = tl.load(q_ptr + b * sq_b + (hk * G + hs)[:, None] * sq_h + t.to(tl.int64) * sq_t + d[None, :],
                    mask=hmask[:, None], other=0.0)
        m_i = tl.full([GP], float("-inf"), tl.float32)
        l_i = tl.zeros([GP], tl.float32)
        acc = tl.zeros([GP, D], tl.float32)
        j = tl.arange(0, NB * R)
        kbase = k_ptr + b * sk_b + hk * sk_h
        vbase = v_ptr + b * sv_b + hk * sv_h
        ibase = idx_ptr + b * si_b + t.to(tl.int64) * si_t
        for i in range(0, TOPK, NB):
            bi = tl.load(ibase + i + j // R)
            ok = bi >= 0
            tok = tl.where(ok, bi * R + j % R, 0)
            acc, m_i, l_i = qsa_attn_step(q, acc, m_i, l_i, kbase, vbase, tok, ok, sk_t, sv_t, d, sm_scale)
        tok = ((pos + 1) // R) * R + j
        ok = (j < R) & (tok <= pos)
        tok = tl.where(ok, tok, 0)
        acc, m_i, l_i = qsa_attn_step(q, acc, m_i, l_i, kbase, vbase, tok, ok, sk_t, sv_t, d, sm_scale)
        out = acc / l_i[:, None]
        tl.store(o_ptr + b * so_b + t.to(tl.int64) * so_t + (hk * G + hs)[:, None] * so_h + d[None, :],
                 out.to(o_ptr.dtype.element_ty), mask=hmask[:, None])


    return (qsa_sparse_attn_kernel,)


@app.cell(hide_code=True)
def _(qsa_sparse_attn_kernel, triton):
    def qsa_sparse_attention(q, k, v, idx, scale, r=4, nb=8, num_warps=4):
        """Sparse attention over indexer-selected blocks.

        q [B, Hq, Tq, D]; k, v [B, Hkv, Tk, D] (the q rows sit at positions Tk - Tq .. Tk - 1); idx [B, Tq, TOPK]
        int32 selected block ids (-1 = none). Each row attends to its selected blocks of r tokens and to the
        tokens of its own incomplete block. Returns [B, Tq, Hq, D] in q.dtype."""
        B, Hq, Tq, D = q.shape
        Hkv, Tk = k.shape[1], k.shape[2]
        G = Hq // Hkv
        topk = idx.shape[-1]
        if topk % nb:
            idx = torch.nn.functional.pad(idx, (0, nb - topk % nb), value=-1)
        idx = idx.contiguous()
        o = torch.empty(B, Tq, Hq, D, dtype=q.dtype, device=q.device)
        qsa_sparse_attn_kernel[(Tq, Hkv, B)](
            q, k, v, idx, o, scale, Tk - Tq,
            q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2), idx.stride(0), idx.stride(1), o.stride(0), o.stride(1), o.stride(2),
            TOPK=idx.shape[-1], G=G, GP=max(16, triton.next_power_of_2(G)), D=D, R=r, NB=nb, num_warps=num_warps)
        return o


    return (qsa_sparse_attention,)


@app.function(hide_code=True)
def qsa_topk_block_ids(indexer, hidden_states, position_embeddings, past_key_values=None, chunk=2048):
    """Indexer selection as block ids instead of a T x T mask: [B, Tq, block_topk] int32, -1 = no block.

    Same scores and the same top-k set as fast_indexer_forward (unpadded causal input). Also appends the
    raw indexer keys to the cache, as the module does."""
    B, Tq, _ = hidden_states.shape
    r, d, Hq = indexer.compress_ratio, indexer.index_head_dim, indexer.index_n_heads
    full_cos, full_sin = position_embeddings
    Tk = full_cos.shape[1]
    q, token_k = torch.split(indexer.index_qk_proj(hidden_states), [Hq * d, d], dim=-1)
    q = indexer.q_layernorm(q.reshape(B, Tq, Hq, d))
    q = qm.apply_rotary_pos_emb(q, cos=full_cos[:, -Tq:, :], sin=full_sin[:, -Tq:, :], unsqueeze_dim=2)
    raw_keys = token_k.reshape(B, Tq, d)
    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, indexer.layer_idx)
    nb = Tk // r
    positions = torch.arange(Tk - Tq, Tk, device=hidden_states.device)
    ids = torch.full((B, Tq, indexer.block_topk), -1, dtype=torch.int32, device=hidden_states.device)
    if nb == 0:
        return ids
    k = min(indexer.block_topk, nb)
    starts = torch.arange(nb, device=hidden_states.device) * r
    for b in range(B):
        kbar = raw_keys[b, : nb * r].reshape(nb, r, d).float().mean(dim=1).to(raw_keys.dtype)
        kbar = indexer.k_layernorm(kbar)
        kbar = qm.apply_rotary_pos_emb(kbar.unsqueeze(1), cos=full_cos[b, starts], sin=full_sin[b, starts]).squeeze(1)
        for c0 in range(0, Tq, chunk):
            sl = slice(c0, min(Tq, c0 + chunk))
            sc = qsa_block_scores(q[b, sl], kbar, positions[sl], r)["score"]
            top = sc.topk(k, dim=-1, sorted=False)
            ids[b, sl, :k] = torch.where(top.values > float("-inf"), top.indices, -1).to(torch.int32)
    return ids


@app.cell(hide_code=True)
def _(qsa_sparse_attention):
    def use_sparse_attention(enabled=True, min_query_len=1024):
        """Route long QSA attention passes (>= min_query_len query rows, no cached prefix, causal mask without
        padding) through qsa_topk_block_ids + qsa_sparse_attention: no T x T masks and no dense attention work.
        Other calls (short suffix passes, cached continuations, padded masks) keep the module's own forward."""
        cls = qm.Qwen4ExpTextAttention
        if "reference_forward" not in cls.__dict__:
            cls.reference_forward = cls.forward
        if not enabled:
            cls.forward = cls.reference_forward
            return False

        def sparse_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
            B, Tq, _ = hidden_states.shape
            plain = (attention_mask is not None and attention_mask.dtype == torch.bool and attention_mask.shape[-1] == Tq
                     and bool(attention_mask[:, 0, -1, :].all()) and not bool(attention_mask[:, 0, 0, 1:].any()))
            if Tq < min_query_len or not plain:
                return cls.reference_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
            ids = qsa_topk_block_ids(self.indexer, hidden_states, position_embeddings, past_key_values)
            cos, sin = (x[:, -Tq:, :] for x in position_embeddings)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)
            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query_states, key_states = qm.apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            attn_output = qsa_sparse_attention(query_states, key_states, value_states, ids, self.scaling,
                                               r=self.indexer.compress_ratio)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous() * torch.sigmoid(gate)
            return self.o_proj(attn_output), None

        cls.forward = sparse_forward
        return True


    return (use_sparse_attention,)


@app.function(hide_code=True)
def causal_conv1d_silu_btc(x, weight, bias=None):
    """Depthwise causal conv1d + SiLU on the [B, T, C] layout: y[t] = silu(sum_j w[:, j] x[t - K + 1 + j] + bias),
    fp32 accumulation. Same as the module's F.conv1d path on the [B, C, T] layout, without the transposes."""
    K, T = weight.shape[-1], x.shape[1]
    xf = F.pad(x, (0, 0, K - 1, 0)).float()
    w = weight.float()
    y = xf[:, 0:T] * w[:, 0]
    for j in range(1, K):
        y = y + xf[:, j : j + T] * w[:, j]
    if bias is not None:
        y = y + bias.float()
    return F.silu(y).to(x.dtype)


@app.function(hide_code=True)
def use_fast_gdn(enabled=True):
    """Gated DeltaNet without the layout copies of the reference forward: a compiled depthwise causal conv1d
    + SiLU on the [B, T, C] layout, grouped value heads passed straight to the fla chunk kernel (16 q/k heads,
    48 v heads, no repeat_interleave), and a compiled gated RMSNorm. Single-token decoding with a cache and
    padded 2-D masks keep the reference forward. Needs use_fla_gdn(True). (The fla causal_conv1d kernel
    autotunes again for every new sequence-length bucket, about 5 s each time, so it is not used.)"""
    import torch._dynamo
    gdn, norm = qm.Qwen4ExpTextGatedDeltaNet, qm.Qwen4ExpTextRMSNormGated
    if "reference_forward" not in gdn.__dict__:
        gdn.reference_forward = gdn.forward
        norm.reference_forward = norm.forward
    if not enabled:
        gdn.forward = gdn.reference_forward
        norm.forward = norm.reference_forward
        return False
    use_fla_gdn(True)
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    if "compiled_forward" not in norm.__dict__:
        norm.compiled_forward = torch.compile(norm.reference_forward, dynamic=None)
    norm.forward = norm.compiled_forward
    if "conv_compiled" not in qm.__dict__:
        qm.conv_compiled = torch.compile(causal_conv1d_silu_btc, dynamic=True)

    def fast_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        B, T, _ = hidden_states.shape
        use_prev = cache_params is not None and cache_params.has_previous_state(self.layer_idx, state_idx=0)
        if ((use_prev and T == 1) or (attention_mask is not None and not bool(attention_mask.all()))
                or self.activation not in ("silu", "swish")):
            return gdn.reference_forward(self, hidden_states, cache_params, attention_mask, **kwargs)
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(B, T, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        if cache_params is not None:
            x = cache_params.update_conv_state(mixed_qkv.transpose(1, 2), self.layer_idx,
                                               conv_kernel_size=self.conv_kernel_size).transpose(1, 2)
        else:
            x = mixed_qkv
        x = qm.conv_compiled(x, self.conv1d.weight.squeeze(1), self.conv1d.bias)[:, -T:]
        query, key, value = torch.split(x, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(B, T, -1, self.head_k_dim)
        key = key.reshape(B, T, -1, self.head_k_dim)
        value = value.reshape(B, T, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_prev else None
        out, last_state = qm.torch_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta, initial_state=state, output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True, cu_seqlens=kwargs.pop("cu_seq_lens_q", None))
        if cache_params is not None:
            cache_params.update_recurrent_state(last_state, self.layer_idx)
        out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(out.reshape(B, T, -1))

    gdn.forward = fast_forward
    return True


@app.function(hide_code=True)
def hc_inject(hyper_input, h, w):
    """Hyper-connection injection of a sublayer output: hyper_input + flatten(h[..., None, :] * w[..., None])."""
    return hyper_input + (h.unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2)


@app.function(hide_code=True)
def moe_combine(out_sorted, inv, weights, T, K):
    """Weighted sum of the K expert outputs of each token; row t*K + k of the (token, slot) order sits at
    out_sorted[inv[t*K + k]]. fp32 accumulation, output in out_sorted.dtype."""
    x = out_sorted[inv].view(T, K, -1).float()
    return (x * weights.float().unsqueeze(-1)).sum(1).to(out_sorted.dtype)


@app.function(hide_code=True)
def silu_mul(h, I):
    """SwiGLU on the fused gate/up output: silu(h[:, :I]) * h[:, I:]."""
    return F.silu(h[:, :I]) * h[:, I:]


@app.function(hide_code=True)
def use_fused_glue(enabled=True):
    """torch.compile the elementwise glue around the big kernels: the hyper-connection injections of every
    decoder layer (one fused kernel instead of a mul and an add on [T, 4 * 2560]), the MoE combine (gather +
    weighted sum instead of index_put + bmm) and SwiGLU. enabled=False restores the reference code."""
    import torch._dynamo
    layer = qm.Qwen4ExpTextDecoderLayer
    if "reference_forward" not in layer.__dict__:
        layer.reference_forward = layer.forward
    if not enabled:
        layer.forward = layer.reference_forward
        NVFP4Experts.combine_fn = None
        NVFP4Experts.silu_mul_fn = None
        return False
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    if "glue_compiled" not in qm.__dict__:
        qm.glue_compiled = dict(inject=torch.compile(hc_inject, dynamic=True),
                                combine=torch.compile(moe_combine, dynamic=True),
                                silu_mul=torch.compile(silu_mul, dynamic=True))
    NVFP4Experts.combine_fn = qm.glue_compiled["combine"]
    NVFP4Experts.silu_mul_fn = qm.glue_compiled["silu_mul"]
    inject = qm.glue_compiled["inject"]

    def fast_forward(self, hidden_states, position_embeddings, attention_mask=None, conv_mask=None,
                     past_key_values=None, ple_input_ids=None, **kwargs):
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(hidden_states, ple_input_ids, past_key_values, conv_mask=conv_mask)
        hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, cache_params=past_key_values, attention_mask=conv_mask, **kwargs)
        else:
            hidden_states, _ = self.self_attn(hidden_states, position_embeddings, attention_mask=attention_mask,
                                              past_key_values=past_key_values, **kwargs)
        hidden_states = inject(hyper_input, hidden_states, injection_weights)
        hidden_states, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return inject(hyper_input, hidden_states, injection_weights)

    layer.forward = fast_forward
    return True


@app.cell
def _(use_grouped_experts, use_sparse_attention):
    def use_fast_path(enabled=True):
        """All speed switches at once: grouped NVFP4 experts, fla GDN kernels with the copy-free GDN forward,
        vectorized indexer, compiled hyper-connections and elementwise glue, FlexAttention for long queries,
        and the QSA sparse-attention kernel for long prefix passes. enabled=False restores the reference path."""
        use_grouped_experts(enabled)
        use_fla_gdn(enabled)
        use_fast_gdn(enabled)
        use_fast_indexer(enabled)
        use_compiled_hyper_connections(enabled)
        use_fused_glue(enabled)
        use_flex_attention(enabled)
        use_sparse_attention(enabled)
        return enabled


    return (use_fast_path,)


@app.cell
def start_pilot_job_fast(extract_teacher_multi):
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

    return (start_pilot_job_fast,)


@app.cell
def _():
    fast_pilot_button = mo.ui.run_button(label="Run fast pilot (both regimes, 3 questions per pass)")
    fast_pilot_button
    return (fast_pilot_button,)


@app.cell
def _(
    fast_pilot_button,
    model,
    pilot_sets,
    start_pilot_job_fast,
    tokenizer,
    use_fast_path,
):
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Phase 2b: standard rerank evaluation (pre-registered 4 Oct 2026, before any teacher pass on these sets)

    **Question.** Can the QSA indexer of Qwen3.8-Flash-Next rerank BM25 candidates as well as the attention heads
    (ICR, QRHead) and standard rerankers? The test sets are held out. The layer and the heads come from the NQ pilot.

    **Data and first stage.**
    - Datasets: TREC DL19 and DL20 passage (judged queries only, 43 and 54), and the BEIR test splits of TREC-COVID (50),
      NFCorpus (323), SciFact (300) and FiQA (648). This gives 1,418 queries.
    - First stage: Pyserini BM25 on the prebuilt flat indexes (k1 = 0.9, b = 0.4), top 100 (`phase2b_bm25.py`).
    - Every reranker reads the same document: `title\ntext`, cut at 256 Qwen tokens (`text_trunc`).

    **Teacher prompt.**
    - Format: the chat prompt (non-thinking), the instruction, then the 100 documents as `[i] document`. Each
      document is padded with newlines to a multiple of 4 tokens, so every document fills whole indexer blocks.
      `Query: <query>` comes last.
    - Document order: a fixed random permutation per query (seed = crc32 of `dataset/qid`), so position and BM25 rank are independent.
    - Rows: the `Query: ...` tokens. Null calibration: the same prompt with the query `N/A`. Calibrated score = mass(query) − mass(null).

    **Primary teacher (fixed).** Indexer, model layer 31 (the best layer on NQ K = 64), softmax over blocks of the
    code-scale score s = I / sqrt(128) at temperature tau = 1. The chunk mass is summed over the document's blocks,
    averaged over the query rows, then null-calibrated.

    **Comparisons (all on the same candidates and the same truncated text).**
    1. BM25 order.
    2. QRHead-16: the sum of the calibrated dense attention masses of the 16 heads with the best NQ K = 64 question-row MRR
       (`PHASE2B["top16_dense"]`).
    3. ICR-288: the sum of the calibrated dense masses of all 288 heads in the 12 softmax-attention layers. The 36 Gated DeltaNet layers have no attention weights.
    4. Supervised cross-encoders: `ms-marco-MiniLM-L-6-v2` and `bge-reranker-v2-m3`.
    5. Dense retrievers used as rerankers: Contriever (unsupervised) and Revela-500M (this repository).
    6. Random order (expected value) and the oracle order (the ceiling of the top-100 candidates).

    **Metric and test.**
    - Metric: nDCG@10 (trec_eval `ndcg_cut_10` through pytrec_eval, graded qrels, ideal DCG over all qrels).
    - Test: paired bootstrap over queries (10,000 resamples, 95% percentile interval), per dataset and for the macro
      mean over the 6 datasets (resampled within each dataset).

    **Decision rules.**
    - H1: the indexer is better than BM25 if the macro difference interval is above 0.
    - H2 and H3 compare the indexer with QRHead-16 and with ICR-288.
      - The indexer is "not worse" at a margin of 0.02 nDCG if the lower bound of the macro interval is above −0.02.
      - The indexer is "worse" if the upper bound is below 0.

    **Secondary analyses** (exploratory, labelled as such):
    - all 12 layers;
    - the temperature grid tau = 2^k / sqrt(128), k = 0..6, plus tau = 1. tau = 1/sqrt(128) is the paper-scale softmax(I) that the indexer was trained on. The pick is made leave-one-dataset-out;
    - raw scores against calibrated scores;
    - mean and max block score;
    - the native top-512 share;
    - sparse-attention heads;
    - the z-scored mean of all layers;
    - position profiles.

    **Amendment (4 Oct 2026).** I made this amendment after the results of DL19, DL20 and 8 TREC-COVID queries were known. Speed is now the main goal, so:
    - every pass stops after layer 35. The primary teacher and QRHead-16 need no later layer, and the forward pass is 25% faster. Without layers 39 to 47, ICR-288 (H3) is not computed;
    - runs use tiers: smoke (DL19 + DL20), dev (+ TREC-COVID and 100 fixed random queries each of NFCorpus, SciFact and FiQA) and full (all 1,418 queries);
    - one baseline (MiniLM) stays as a sanity check.

    The primary teacher, H1, H2 and the tests do not change.
    """)

    return


@app.cell(hide_code=True)
def _():
    PHASE2B = dict(
        cand_dir="/root/models/phase2b",
        datasets=["dl19", "dl20", "trec-covid", "nfcorpus", "scifact", "fiqa"],
        k=100,
        max_doc_tokens=256,
        temps=sorted([2 ** k / math.sqrt(128) for k in range(7)] + [1.0]),
        primary_temp=1.0,
        primary_layer=31,
        qsa_layers=[3 + 4 * i for i in range(12)],
        # (model layer, head): the 16 best dense heads on NQ K = 64 question rows, null-calibrated (results/pilot_scores_k64.csv)
        top16_dense=[(31, 12), (27, 2), (31, 2), (35, 13), (27, 19), (31, 5), (31, 3), (27, 3),
                     (31, 15), (31, 16), (27, 0), (35, 2), (35, 12), (35, 10), (35, 18), (31, 14)],
        n_boot=10000,
        margin=0.02,
        baselines=["ce-minilm"],  # one supervised reranker as a sanity check of the candidates and the eval code
        last_layer=35,  # stop every pass after layer 35: the primary indexer (31) and the QRHead heads (27, 31, 35) need no later layer
        # evaluation tiers: datasets and the fixed random subset size per dataset (None = all queries)
        tiers={
            "smoke": dict(datasets=["dl19", "dl20"], max_queries={}),
            "dev": dict(datasets=["dl19", "dl20", "trec-covid", "nfcorpus", "scifact", "fiqa"],
                        max_queries={"nfcorpus": 100, "scifact": 100, "fiqa": 100}),
            "full": dict(datasets=["dl19", "dl20", "trec-covid", "nfcorpus", "scifact", "fiqa"], max_queries={}),
        },
    )

    return (PHASE2B,)


@app.function(hide_code=True)
def build_rerank_example(tokenizer, query, docs, r=4):
    """Chat-format rerank prompt (ICR-style context): instruction, numbered documents, then the query.

    docs: list of document strings (already truncated). Each document chunk "[i] doc" is padded with
    newlines to a multiple of r tokens (at least one newline), so chunk spans are block-aligned.
    Returns ids [1, T], chunk spans, the query rows ("Query: ..." tokens) and empty answer rows."""
    def enc(s):
        return tokenizer(s, add_special_tokens=False).input_ids
    nl = enc("\n")[0]
    pre = enc("<|im_start|>user\nRead the documents, then find the documents that are relevant to the query.\n\n")
    pre = pre + [nl] * (-len(pre) % r)
    spans, body = [], []
    for i, d in enumerate(docs):
        c = enc(f"[{i + 1}] {d}")
        c = c + [nl] * (1 + (-(len(c) + 1)) % r)
        start = len(pre) + len(body)
        spans.append((start, start + len(c)))
        body += c
    q = enc(f"Query: {query}")
    mid = enc("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    ids = pre + body + q + mid
    q0 = len(pre) + len(body)
    return dict(ids=torch.tensor([ids]), spans=spans, q_rows=torch.arange(q0, q0 + len(q)),
                a_rows=torch.arange(0))


@app.function(hide_code=True)
def segment_sum(w, seg, n):
    """w [..., T]; seg [T] chunk index in 0..n (n = outside every chunk) -> [..., n] summed weight."""
    out = torch.zeros(*w.shape[:-1], n + 1, dtype=torch.float32, device=w.device)
    out.index_add_(w.dim() - 1, seg, w.float())
    return out[..., :n]


@app.function(hide_code=True)
def rerank_signals_from_store(model, store, ex, temps):
    """Chunk-level teacher signals for the query rows of a build_rerank_example prompt.

    Returns CPU float tensors (L = 12 QSA layers, K = documents, H = 24 heads):
      idx    [L, len(temps) + 2, K]: indexer softmax mass sum_{b in chunk} softmax(s / tau)_b, averaged
                                     over query rows, for every tau in temps (s = code-scale score
                                     I / sqrt(128); tau = 1/sqrt(128) is the paper-scale softmax(I)); then
                                     the mean and the max block score in the chunk
      sel    [L, K]:    share of the chunk's blocks inside the native top-512 selection
      dense  [L, H, K]: dense attention mass per head; sparse: same under the indexer selection
    """
    device = model.lm_head.weight.device
    spans = ex["spans"]
    K = len(spans)
    rows = ex["q_rows"].to(device)
    T = int(ex["ids"].shape[1])
    seg_tok = torch.full((T,), K, dtype=torch.long, device=device)
    for k, (s, e) in enumerate(spans):
        seg_tok[s:e] = k
    out = {k: [] for k in ("idx", "sel", "dense", "sparse")}
    for li in sorted(store):
        x, (cos, sin) = store[li]
        x, cos, sin = x[0], cos[0], sin[0]
        attn = model.model.layers[li].self_attn
        r = attn.indexer.compress_ratio
        tok, sc, selb = indexer_selection_tokens(attn.indexer, x, cos, sin, rows)
        full, valid = sc["score"], sc["valid"]
        nb = full.shape[-1]
        seg_blk = seg_tok[: nb * r : r]
        masses = [segment_sum((full / t).softmax(-1), seg_blk, K).mean(0) for t in temps]
        finite = full.masked_fill(~valid, 0.0)
        cnt = segment_sum(valid.float(), seg_blk, K).clamp_min(1)
        mean_sc = (segment_sum(finite, seg_blk, K) / cnt).mean(0)
        max_sc = torch.stack([full[:, s // r : e // r].amax(-1) for s, e in spans], -1).mean(0)
        out["idx"].append(torch.stack(masses + [mean_sc, max_sc]))
        out["sel"].append((segment_sum(selb.float(), seg_blk, K) / cnt).mean(0))
        de = attention_side_path(attn, x, cos, sin, rows, None)
        out["dense"].append(segment_sum(de["probs"], seg_tok, K).mean(0))
        del de
        sp = attention_side_path(attn, x, cos, sin, rows, tok)
        out["sparse"].append(segment_sum(sp["probs"], seg_tok, K).mean(0))
        del sp
    return {k: torch.stack(v).float().cpu() for k, v in out.items()}


@app.function(hide_code=True)
def segment_max(w, seg, n):
    """w [..., T]; seg [T] chunk index in 0..n (n = outside every chunk) -> [..., n] max (-inf for an empty chunk)."""
    out = torch.full((*w.shape[:-1], n + 1), float("-inf"), dtype=torch.float32, device=w.device)
    out.scatter_reduce_(w.dim() - 1, seg.expand(w.shape).contiguous(), w.float(), reduce="amax", include_self=True)
    return out[..., :n]


@app.function(hide_code=True)
def attention_logits_side(attn, x, cos, sin, rows):
    """Dense attention logits of all heads for the query rows and the value-output norm of every token.

    Returns logits [R, Hq, T] fp32 (-inf after the row, scaled as in the module) and vnorm [T, Hq] =
    ||v_t W_O^h|| (the vector that head h writes into the residual stream per unit of attention on token t,
    before the output gate)."""
    T, dh = x.shape[0], attn.head_dim
    query, _ = torch.chunk(attn.q_proj(x[rows]).view(rows.numel(), -1, dh * 2), 2, dim=-1)
    q = qm.apply_rotary_pos_emb(attn.q_norm(query), cos=cos[rows], sin=sin[rows], unsqueeze_dim=1)
    k = qm.apply_rotary_pos_emb(attn.k_norm(attn.k_proj(x).view(T, -1, dh)), cos=cos, sin=sin, unsqueeze_dim=1)
    v = attn.v_proj(x).view(T, -1, dh)
    Hq, rep = q.shape[1], q.shape[1] // k.shape[1]
    logits = torch.einsum("rhd,thd->rht", q.float(), k.repeat_interleave(rep, dim=1).float()) * attn.scaling
    logits = logits.masked_fill(torch.arange(T, device=x.device)[None, None, :] > rows[:, None, None], float("-inf"))
    w_o = attn.o_proj.weight
    vnorm = torch.stack([(v[:, h // rep] @ w_o[:, h * dh : (h + 1) * dh].T).float().norm(dim=-1) for h in range(Hq)], -1)
    return logits, vnorm


@app.function(hide_code=True)
def rerank_maxsim_signals(model, store, ex, temps):
    """rerank_signals_from_store plus late-interaction (MaxSim) and value-weighted signals, query rows only.

    Extra keys (L QSA layers, K documents, H = 4 indexer heads, Hq = 24 attention heads):
      ms      [L, 6, K]: indexer MaxSim, mean over query rows of max over the document's blocks of
                         0 sum_h ReLU(q_h . k_b) (the score I), 1 sum_h max_b ReLU(q_h . k_b) (per head),
                         2 sum_h max_b (q_h . k_b) on the position-free dims, 3 sum_h max_b cos(q_h, k_b),
                         4 the same cosine on the position-free dims, 5 max_b of I z-normalized per row
      ms_head [L, 3, H, K]: per indexer head: max dot, max cosine, max position-free cosine
      att_ms  [L, 2, Hq, K]: per attention head: max logit over the document's tokens, and the same after
                         z-normalizing each row's logits
      att_vw  [L, Hq, K]: value-weighted attention mass: sum over the document of a_t ||v_t W_O^h||,
                         normalized per row and head"""
    out = rerank_signals_from_store(model, store, ex, temps)
    device = model.lm_head.weight.device
    spans, K = ex["spans"], len(ex["spans"])
    rows = ex["q_rows"].to(device)
    T = int(ex["ids"].shape[1])
    seg_tok = torch.full((T,), K, dtype=torch.long, device=device)
    for k, (s, e) in enumerate(spans):
        seg_tok[s:e] = k
    extra = {k: [] for k in ("ms", "ms_head", "att_ms", "att_vw")}
    for li in sorted(store):
        x, (cos, sin) = store[li]
        x, cos, sin = x[0], cos[0], sin[0]
        attn = model.model.layers[li].self_attn
        r, rd = attn.indexer.compress_ratio, cos.shape[-1]
        q, kbar = qsa_index_parts(attn.indexer, x, cos, sin)
        qr, kf = q[rows].float(), kbar.float()
        nb = kf.shape[0]
        seg_blk = seg_tok[: nb * r : r]
        valid = torch.arange(nb, device=device)[None, :] < ((rows + 1) // r)[:, None]
        neg = float("-inf")
        dots = torch.einsum("rhd,bd->rhb", qr, kf)
        nope = torch.einsum("rhd,bd->rhb", qr[..., rd:], kf[:, rd:])
        cosf = torch.einsum("rhd,bd->rhb", F.normalize(qr, dim=-1), F.normalize(kf, dim=-1))
        cosn = torch.einsum("rhd,bd->rhb", F.normalize(qr[..., rd:], dim=-1), F.normalize(kf[:, rd:], dim=-1))
        vm = valid[:, None, :]
        I = torch.relu(dots).sum(1).masked_fill(~valid, neg)
        Iv = I.masked_fill(~valid, 0.0)
        cnt = valid.sum(-1, keepdim=True).clamp_min(1)
        mu = Iv.sum(-1, keepdim=True) / cnt
        sd = (((Iv - mu) ** 2) * valid).sum(-1, keepdim=True).div(cnt).sqrt().clamp_min(1e-6)
        Iz = ((I - mu) / sd).masked_fill(~valid, neg)
        per_head = [segment_max(t.masked_fill(~vm, neg), seg_blk, K) for t in (torch.relu(dots), nope, cosf, cosn)]
        ms = torch.stack([segment_max(I, seg_blk, K).mean(0)] + [p.sum(1).mean(0) for p in per_head]
                         + [segment_max(Iz, seg_blk, K).mean(0)])
        extra["ms"].append(ms)
        extra["ms_head"].append(torch.stack([per_head[0].mean(0), per_head[2].mean(0), per_head[3].mean(0)]))
        logits, vnorm = attention_logits_side(attn, x, cos, sin, rows)
        fin = torch.isfinite(logits)
        lz = logits.masked_fill(~fin, 0.0)
        n = fin.sum(-1, keepdim=True).clamp_min(1)
        lmu = lz.sum(-1, keepdim=True) / n
        lsd = (((lz - lmu) ** 2) * fin).sum(-1, keepdim=True).div(n).sqrt().clamp_min(1e-6)
        extra["att_ms"].append(torch.stack([segment_max(logits, seg_tok, K).mean(0),
                                            segment_max(((logits - lmu) / lsd).masked_fill(~fin, neg), seg_tok, K).mean(0)]))
        w = logits.softmax(-1) * vnorm.T[None]
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
        extra["att_vw"].append(segment_sum(w, seg_tok, K).mean(0))
        del logits, lz, w, dots, nope, cosf, cosn
    out.update({k: torch.stack(v).float().cpu() for k, v in extra.items()})
    return out


@app.function(hide_code=True)
@torch.no_grad()
def query_likelihood_scores(model, tokenizer, query, docs, batch_tokens=16000, row_chunk=8):
    """UPR-style relevance: mean log p(query token | document, instruction) over the query tokens, one short
    sequence per document, batched with right padding (causal, so real rows never see the padding).

    Sequence: chat prompt "Document: {doc}\\n\\nWrite a search query for this document." then the assistant
    turn (non-thinking) with the query. Returns a tensor [len(docs)] (higher = more relevant). A null-document
    calibration would subtract one constant per query and does not change the ranking."""
    device = model.lm_head.weight.device
    def enc(s):
        return tokenizer(s, add_special_tokens=False).input_ids
    head = enc("<|im_start|>user\nDocument: ")
    tail = enc("\n\nWrite a search query for this document.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    q = enc(query)
    seqs = [head + enc(d) + tail + q for d in docs]
    scores = torch.empty(len(docs))
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * len(seqs[order[j]]) <= batch_tokens:
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        L = max(len(seqs[k]) for k in idx)
        ids = torch.full((len(idx), L), 198, dtype=torch.long)
        for r, k in enumerate(idx):
            ids[r, : len(seqs[k])] = torch.tensor(seqs[k])
        h = model.model(ids.to(device), use_cache=False).last_hidden_state
        ends = torch.tensor([len(seqs[k]) for k in idx], device=device)
        pos = ends[:, None] - len(q) - 1 + torch.arange(len(q), device=device)[None, :]
        hq = h[torch.arange(len(idx), device=device)[:, None], pos]
        tgt = torch.tensor(q, device=device)
        for c0 in range(0, len(idx), row_chunk):
            # cross-entropy over a few rows at a time: no full-vocabulary log-softmax copy for long queries
            lg = model.lm_head(hq[c0 : c0 + row_chunk]).float()
            nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.repeat(lg.shape[0]), reduction="none").view(lg.shape[0], -1)
            scores[torch.tensor(idx[c0 : c0 + row_chunk])] = (-nll.mean(-1)).cpu()
            del lg, nll
        del h, hq
        i = j
    return scores


@app.function(hide_code=True)
def start_upr_job(model, tokenizer, datasets, cand_dir, max_queries=None, tag="upr"):
    """Query-likelihood scores (query_likelihood_scores) for the BM25 candidates of every query, in a background
    thread. Saves <cand_dir>/run_<tag>_<dataset>.pt as {qid: {docid: score}} (the format of the baseline runs)."""
    import threading, traceback
    thread_name = f"qsa-upr-{tag}"
    if any(t.name == thread_name and t.is_alive() for t in threading.enumerate()):
        raise RuntimeError(f"a {thread_name} job is already running")
    state = {"status": "running", "progress": "", "error": None, "t0": time.time(), "per_dataset_s": {}, "stop": False}

    def work():
        try:
            for name in datasets:
                data = torch.load(os.path.join(cand_dir, f"candidates_{name}.pt"), weights_only=False)
                path = os.path.join(cand_dir, f"run_{tag}_{name}.pt")
                run = torch.load(path, weights_only=False) if os.path.exists(path) else {}
                t_ds = time.time()
                qids = phase2b_subset(name, data["queries"], (max_queries or {}).get(name))
                for n_done, qid in enumerate(qids):
                    if state["stop"]:
                        torch.save(run, path)
                        state["status"] = "stopped"
                        return
                    if qid in run:
                        continue
                    docids = [d for d, _ in data["cands"][qid]]
                    if docids:
                        sc = query_likelihood_scores(model, tokenizer, data["queries"][qid], [data["docs"][d]["text_trunc"] for d in docids])
                        run[qid] = dict(zip(docids, sc.tolist()))
                    else:
                        run[qid] = {}
                    if len(run) % 25 == 0:
                        torch.save(run, path)
                    state["progress"] = f"{name} {n_done + 1}/{len(qids)} ({time.time() - state['t0']:.0f}s)"
                torch.save(run, path)
                state["per_dataset_s"][name] = time.time() - t_ds
            state["status"] = "finished"
        except Exception:
            state["status"] = "error"
            state["error"] = traceback.format_exc()

    state["thread"] = threading.Thread(target=work, name=thread_name, daemon=True)
    state["thread"].start()
    return state


@app.function(hide_code=True)
def indexer_head_mass(indexer, x, cos, sin, rows, seg_blk, K, temps=(1.0, 1.414)):
    """Softmax mass per indexer head: for head h, softmax over blocks of ReLU(q_h . k_b) / sqrt(d) / tau, summed
    over each document's blocks and averaged over rows. Returns [H, len(temps), K]."""
    r, d = indexer.compress_ratio, indexer.index_head_dim
    q, kbar = qsa_index_parts(indexer, x, cos, sin)
    s = torch.relu(torch.einsum("rhd,bd->rhb", q[rows].float(), kbar.float())) / math.sqrt(d)
    valid = torch.arange(kbar.shape[0], device=x.device)[None, :] < ((rows + 1) // r)[:, None]
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    return torch.stack([segment_sum((s / t).softmax(-1), seg_blk, K).mean(0) for t in temps], 1)


@app.function(hide_code=True)
def rerank_rows_signals(model, store, ex, temps):
    """Signals for two row sets: the query tokens (keys as in rerank_signals_from_store) and the decision-point
    rows after the query (the end-of-turn and assistant-header tokens, keys prefixed with "mid_"), plus the
    per-indexer-head softmax mass for both row sets ("ihead" [L, 2 row sets, H, 2 taus, K], taus 1 and 1.414)."""
    T = int(ex["ids"].shape[1])
    q_end = int(ex["q_rows"][-1]) + 1
    ex_mid = dict(ex, q_rows=torch.arange(q_end, T))
    out = rerank_signals_from_store(model, store, ex, temps)
    out.update({f"mid_{k}": v for k, v in rerank_signals_from_store(model, store, ex_mid, temps).items()})
    device = model.lm_head.weight.device
    K = len(ex["spans"])
    seg_tok = torch.full((T,), K, dtype=torch.long, device=device)
    for k, (s, e) in enumerate(ex["spans"]):
        seg_tok[s:e] = k
    ih = []
    for li in sorted(store):
        x, (cos, sin) = store[li]
        x, cos, sin = x[0], cos[0], sin[0]
        indexer = model.model.layers[li].self_attn.indexer
        nb = x.shape[0] // indexer.compress_ratio
        seg_blk = seg_tok[: nb * indexer.compress_ratio : indexer.compress_ratio]
        ih.append(torch.stack([indexer_head_mass(indexer, x, cos, sin, rr.to(device), seg_blk, K)
                               for rr in (ex["q_rows"], ex_mid["q_rows"])]))
    out["ihead"] = torch.stack(ih).float().cpu()
    return out


@app.function(hide_code=True)
def build_congress_pool(tokenizer, base="/root/models/obliq/tip-of-tongue/congress", k=50, max_doc_tokens=1024,
                        out="/root/models/phase2b/candidates_congress.pt"):
    """Congress Hearings (OBLIQ-Bench) verification pool: the gold passage plus k - 1 hard negatives, first from
    the gold's own hearing (same people, topic and vocabulary), then from the other hearings that hold golds.
    Random choices use crc32(qid) seeds. Saved in the candidates_<name>.pt format (cands in random order with
    score 0, since there is no first-stage ranking). base: the congress folder of the dianetc/OBLIQ-Bench
    dataset (tip-of-tongue/congress). Returns pool statistics."""
    import json, zlib
    queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(f"{base}/queries+qrels/queries.jsonl")}
    qrels = {}
    for line in open(f"{base}/queries+qrels/qrels.tsv"):
        q, d, s = line.rstrip("\n").split("\t")
        if q != "query-id":
            qrels[q] = {d: int(s)}
    hearing = lambda d: d.split("_p")[0]
    gold_h = {hearing(d) for v in qrels.values() for d in v}
    by_h, text = {}, {}
    for line in open(f"{base}/corpus/corpus.jsonl"):
        r = json.loads(line)
        h = hearing(r["_id"])
        if h in gold_h:
            by_h.setdefault(h, []).append(r["_id"])
            text[r["_id"]] = r["text"]
    other_pool = sorted(text)
    cands, n_same = {}, []
    for q in sorted(qrels):
        gold = next(iter(qrels[q]))
        g = torch.Generator().manual_seed(zlib.crc32(f"congress/{q}".encode()))
        same = [d for d in by_h[hearing(gold)] if d != gold]
        same = [same[i] for i in torch.randperm(len(same), generator=g)[: k - 1].tolist()]
        rest = [d for d in other_pool if hearing(d) != hearing(gold)]
        fill = [rest[i] for i in torch.randperm(len(rest), generator=g)[: k - 1 - len(same)].tolist()]
        pool = [gold] + same + fill
        pool = [pool[i] for i in torch.randperm(len(pool), generator=g).tolist()]
        cands[q] = [(d, 0.0) for d in pool]
        n_same.append(len(same))
    docs = {}
    for q, c in cands.items():
        for d, _ in c:
            if d not in docs:
                ids = tokenizer(" ".join(text[d].split()), add_special_tokens=False).input_ids[:max_doc_tokens]
                docs[d] = dict(text=text[d], ids=ids, text_trunc=tokenizer.decode(ids))
    data = dict(name="congress", index="obliq-congress-same-hearing-pool", k=k, max_doc_tokens=max_doc_tokens,
                queries={q: queries[q] for q in qrels}, qrels=qrels, cands=cands, docs=docs)
    torch.save(data, out)
    gold_tok = sorted(len(docs[next(iter(qrels[q]))]["ids"]) for q in qrels)
    cut = sum(len(tokenizer(" ".join(text[next(iter(qrels[q]))].split()), add_special_tokens=False).input_ids) > max_doc_tokens for q in qrels)
    qtok = sorted(len(tokenizer(queries[q], add_special_tokens=False).input_ids) for q in qrels)
    return dict(queries=len(qrels), mean_same_hearing_negs=round(sum(n_same) / len(n_same), 1), min_same=min(n_same),
                docs=len(docs), gold_tokens_median=gold_tok[len(gold_tok) // 2], golds_cut=cut, query_tokens_median=qtok[len(qtok) // 2])


@app.function(hide_code=True)
def build_congress_dense_pool(tokenizer, run_paths, base="/root/models/obliq/tip-of-tongue/congress", size=300, max_doc_tokens=1024,
                              out="/root/models/phase2b/candidates_congress300.pt"):
    """OBLIQ-Bench Congress pool in the style of the paper's GPT-5.2 oracle (Sec. 5): the union of the top results
    of the first-stage runs in run_paths ({qid: [(docid, score), ...]}), merged in turn (rank 1 of each run,
    then rank 2, ...) until size - 1 distinct non-gold passages, plus the gold passage, in a fixed random
    order (crc32 seed). The paper also used Gemini-Embedding-2, which is not available here.
    Returns pool statistics, including how often the first stage found the gold inside the pool."""
    import json, zlib
    queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(f"{base}/queries+qrels/queries.jsonl")}
    qrels = {}
    for line in open(f"{base}/queries+qrels/qrels.tsv"):
        q, d, s = line.rstrip("\n").split("\t")
        if q != "query-id":
            qrels[q] = {d: int(s)}
    runs = [torch.load(p, weights_only=False) for p in run_paths]
    cands, found = {}, 0
    for q in sorted(qrels):
        gold = next(iter(qrels[q]))
        pool, seen, r = [], {gold}, 0
        in_first_stage = False
        while len(pool) < size - 1:
            added = False
            for run in runs:
                if r < len(run[q]):
                    d = run[q][r][0]
                    added = True
                    if d == gold:
                        in_first_stage = True
                    elif d not in seen and len(pool) < size - 1:
                        pool.append(d)
                        seen.add(d)
            if not added:
                break
            r += 1
        found += in_first_stage
        pool = pool + [gold]
        g = torch.Generator().manual_seed(zlib.crc32(f"congress300/{q}".encode()))
        pool = [pool[i] for i in torch.randperm(len(pool), generator=g).tolist()]
        cands[q] = [(d, 0.0) for d in pool]
    need = {d for c in cands.values() for d, _ in c}
    text = {}
    for line in open(f"{base}/corpus/corpus.jsonl"):
        rec = json.loads(line)
        if rec["_id"] in need:
            text[rec["_id"]] = rec["text"]
    docs = {}
    for d in need:
        ids = tokenizer(" ".join(text[d].split()), add_special_tokens=False).input_ids[:max_doc_tokens]
        docs[d] = dict(text=text[d], ids=ids, text_trunc=tokenizer.decode(ids))
    data = dict(name="congress300", index="obliq-congress-dense-union-pool", k=size, max_doc_tokens=max_doc_tokens,
                queries={q: queries[q] for q in qrels}, qrels=qrels, cands=cands, docs=docs)
    torch.save(data, out)
    sizes = sorted(len(c) for c in cands.values())
    return dict(queries=len(qrels), pool_size_min=sizes[0], pool_size_median=sizes[len(sizes) // 2], unique_docs=len(docs),
                gold_found_by_first_stage=round(found / len(qrels), 3))


@app.function(hide_code=True)
def rerank_doc_order(dataset, qid, n):
    """Fixed random permutation of the n BM25 candidates for one query (seed = crc32 of 'dataset/qid')."""
    import zlib
    g = torch.Generator().manual_seed(zlib.crc32(f"{dataset}/{qid}".encode()))
    return torch.randperm(n, generator=g).tolist()


@app.function(hide_code=True)
def phase2b_subset(dataset, qids, n=None):
    """Fixed random subset of n query ids, sorted (all ids when n is None or n >= len(qids)); seed = crc32(dataset)."""
    import zlib
    qids = sorted(qids)
    if n is None or n >= len(qids):
        return qids
    g = torch.Generator().manual_seed(zlib.crc32(dataset.encode()))
    return sorted(qids[i] for i in torch.randperm(len(qids), generator=g)[:n].tolist())


@app.cell(hide_code=True)
def _(extract_teacher_multi):
    def start_rerank_job(model, tokenizer, datasets, cand_dir, temps, out_dir=None, tag="qsa", last_layer=None, max_queries=None,
                         signals_fn=None):
        """Phase 2b teacher extraction in a background thread. For every query: the BM25 candidates in a fixed
        random order, one prefix pass, then the query and the null query "N/A" as one suffix batch. Results go
        to <out_dir>/teacher_<tag>_<dataset>.pt as {qid: dict(docids, q, null)}, saved every 25 queries so
        the job can resume. A query without BM25 candidates gets docids = [] and q = null = None.
        last_layer: stop the passes after this decoder layer (signals only for the QSA layers up to it).
        max_queries: {dataset: n} runs only the fixed subset phase2b_subset(dataset, qids, n).
        signals_fn(model, store, ex): the signals of one prompt variant (default: rerank_signals_from_store
        with temps; rerank_maxsim_signals and rerank_rows_signals add more signals)."""
        import threading, traceback
        thread_name = f"qsa-rerank-{tag}"
        if any(t.name == thread_name and t.is_alive() for t in threading.enumerate()):
            raise RuntimeError(f"a {thread_name} job is already running; two jobs would share the GPU and the output files")
        out_dir = out_dir or cand_dir
        fn = signals_fn or (lambda m, s, e: rerank_signals_from_store(m, s, e, temps))
        state = {"status": "running", "progress": "", "error": None, "t0": time.time(), "per_dataset_s": {}, "stop": False}

        def work():
            try:
                for name in datasets:
                    data = torch.load(os.path.join(cand_dir, f"candidates_{name}.pt"), weights_only=False)
                    path = os.path.join(out_dir, f"teacher_{tag}_{name}.pt")
                    recs = torch.load(path, weights_only=False) if os.path.exists(path) else {}
                    t_ds = time.time()
                    qids = phase2b_subset(name, data["queries"], (max_queries or {}).get(name))
                    for n_done, qid in enumerate(qids):
                        if state["stop"]:
                            torch.save(recs, path)
                            state["status"] = "stopped"
                            return
                        if qid in recs:
                            continue
                        cands = [d for d, _ in data["cands"][qid]]
                        if not cands:
                            recs[qid] = dict(docids=[], q=None, null=None, n_tokens=0)
                            continue
                        order = rerank_doc_order(name, qid, len(cands))
                        docids = [cands[i] for i in order]
                        texts = [data["docs"][d]["text_trunc"] for d in docids]
                        group = [build_rerank_example(tokenizer, q, texts) for q in (data["queries"][qid], "N/A")]
                        sig_q, sig_null = extract_teacher_multi(
                            model, [group], signals_fn=fn, last_layer=last_layer)[0]
                        recs[qid] = dict(docids=docids, q=sig_q, null=sig_null, n_tokens=int(group[0]["ids"].shape[1]))
                        if len(recs) % 25 == 0:
                            torch.save(recs, path)
                        state["progress"] = f"{name} {n_done + 1}/{len(qids)} ({time.time() - state['t0']:.0f}s)"
                    torch.save(recs, path)
                    state["per_dataset_s"][name] = time.time() - t_ds
                state["status"] = "finished"
            except Exception:
                state["status"] = "error"
                state["error"] = traceback.format_exc()

        state["thread"] = threading.Thread(target=work, name=thread_name, daemon=True)
        state["thread"].start()
        return state


    return (start_rerank_job,)


@app.function(hide_code=True)
def trec_ndcg10(qrels, run):
    """Per-query nDCG@10 with trec_eval semantics (pytrec_eval ndcg_cut_10). Queries of qrels that are
    missing from run score 0."""
    import pytrec_eval
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    res = ev.evaluate({q: {d: float(s) for d, s in run[q].items()} for q in qrels if q in run})
    return {q: res[q]["ndcg_cut_10"] if q in res else 0.0 for q in qrels}


@app.function(hide_code=True)
def paired_bootstrap(diffs, n_boot=10000, seed=0):
    """diffs: list of per-dataset 1-D tensors of per-query differences (system A - system B).
    Resamples queries within each dataset; the statistic is the mean over datasets of the per-dataset
    mean. Returns (point estimate, 2.5% quantile, 97.5% quantile)."""
    g = torch.Generator().manual_seed(seed)
    point = torch.stack([d.mean() for d in diffs]).mean().item()
    boot = torch.zeros(n_boot)
    for d in diffs:
        idx = torch.randint(0, d.numel(), (n_boot, d.numel()), generator=g)
        boot += d[idx].mean(1)
    boot /= len(diffs)
    lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975])).tolist()
    return point, lo, hi


@app.function(hide_code=True)
def phase2b_teachers(cfg, n_layers=12):
    """Teacher scorers for Phase 2b: name -> fn(sig_q, sig_null) -> scores [K].

    The primary and the pre-registered comparison teachers come first. Index of a layer = position in cfg["qsa_layers"];
    idx[:, j] = temperature j of cfg["temps"], then mean score (-2) and max score (-1).
    n_layers: number of QSA layers in the records (9 when the passes stop after layer 35). A teacher is
    built only when all of its layers are present (ICR-288 needs all 12)."""
    L = cfg["qsa_layers"][:n_layers]
    li = L.index(cfg["primary_layer"])
    ti = cfg["temps"].index(cfg["primary_temp"])
    heads = [(L.index(l), h) for l, h in cfg["top16_dense"]]
    hl = torch.tensor([a for a, _ in heads])
    hh = torch.tensor([b for _, b in heads])
    t = {
        "indexer_L31_t1_cal": lambda q, n: q["idx"][li, ti] - n["idx"][li, ti],
        "qrhead16_cal": lambda q, n: (q["dense"][hl, hh] - n["dense"][hl, hh]).sum(0),
        "indexer_L31_t1_raw": lambda q, n: q["idx"][li, ti],
        "qrhead16_raw": lambda q, n: q["dense"][hl, hh].sum(0),
        "qrhead16_sparse_cal": lambda q, n: (q["sparse"][hl, hh] - n["sparse"][hl, hh]).sum(0),
    }
    if n_layers == 12:
        t["icr288_cal"] = lambda q, n: (q["dense"] - n["dense"]).sum((0, 1))
        t["icr288_raw"] = lambda q, n: q["dense"].sum((0, 1))
        t["icr288_sparse_cal"] = lambda q, n: (q["sparse"] - n["sparse"]).sum((0, 1))

    def z(x):
        return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-12)

    t["indexer_zmean_t1_cal"] = lambda q, n: z(q["idx"][:n_layers, ti] - n["idx"][:n_layers, ti]).mean(0)
    for a, layer in enumerate(L):
        for j, tau in enumerate(cfg["temps"]):
            t[f"indexer_L{layer}_tau{tau:.3f}_cal"] = lambda q, n, a=a, j=j: q["idx"][a, j] - n["idx"][a, j]
            t[f"indexer_L{layer}_tau{tau:.3f}_raw"] = lambda q, n, a=a, j=j: q["idx"][a, j]
        t[f"indexer_L{layer}_meanscore_cal"] = lambda q, n, a=a: q["idx"][a, -2] - n["idx"][a, -2]
        t[f"indexer_L{layer}_maxscore_cal"] = lambda q, n, a=a: q["idx"][a, -1] - n["idx"][a, -1]
        t[f"native_sel_L{layer}"] = lambda q, n, a=a: q["sel"][a]
        t[f"dense_layer_L{layer}_cal"] = lambda q, n, a=a: (q["dense"][a] - n["dense"][a]).sum(0)
    return t


@app.function(hide_code=True)
def phase2b_evaluate(cfg, teacher_recs, tag="qsa"):
    """Per-query nDCG@10 of every system on every dataset that has teacher records.

    teacher_recs {dataset: {qid: dict(docids, q, null)}}. Returns (per_query {system: {dataset: tensor}},
    qids {dataset: list}). Systems: bm25, random (mean of 20 permutations), oracle, the baselines that
    have runs, and every teacher of phase2b_teachers. Only queries with teacher records are scored; a query
    without candidates scores 0 for every system."""
    n_layers = min(r["q"]["idx"].shape[0] for recs in teacher_recs.values() for r in recs.values() if r["q"] is not None)
    teachers = phase2b_teachers(cfg, n_layers)
    per_query, qids_by_ds = {}, {}
    for ds, recs in teacher_recs.items():
        data = torch.load(os.path.join(cfg["cand_dir"], f"candidates_{ds}.pt"), weights_only=False)
        qids = sorted(q for q in data["queries"] if q in recs)
        qids_by_ds[ds] = qids
        qrels = {q: data["qrels"][q] for q in qids}

        def add(name, run):
            nd = trec_ndcg10(qrels, run)
            per_query.setdefault(name, {})[ds] = torch.tensor([nd[q] for q in qids])

        add("bm25", {q: {d: -i for i, (d, _) in enumerate(data["cands"][q])} for q in qids})
        add("oracle", {q: {d: qrels[q].get(d, 0) - i / 1000 for i, (d, _) in enumerate(data["cands"][q])} for q in qids})
        g = torch.Generator().manual_seed(0)
        rand = []
        for _ in range(20):
            nd = trec_ndcg10(qrels, {q: {d: float(s) for (d, _), s in zip(data["cands"][q], torch.rand(len(data["cands"][q]), generator=g).tolist())}
                                     for q in qids})
            rand.append(torch.tensor([nd[q] for q in qids]))
        per_query.setdefault("random", {})[ds] = torch.stack(rand).mean(0)
        for b in cfg["baselines"]:
            p = os.path.join(cfg["cand_dir"], f"run_{b}_{ds}.pt")
            if os.path.exists(p):
                add(b, torch.load(p, weights_only=False))
        for name, fn in teachers.items():
            add(name, {q: dict(zip(recs[q]["docids"], fn(recs[q]["q"], recs[q]["null"]).nan_to_num(0.0).tolist())) if recs[q]["docids"] else {}
                       for q in qids})
    return per_query, qids_by_ds


@app.function(hide_code=True)
def phase2b_report(cfg, per_query, primary="indexer_L31_t1_cal", against=None):
    """(table, tests): table = nDCG@10 per system and dataset plus the macro mean; tests = paired
    bootstrap of the primary teacher against the systems in `against` (default: the pre-registered
    comparisons and the baselines), macro and per dataset."""
    against = against or ["bm25", "qrhead16_cal", "icr288_cal", "random", "oracle"] + cfg["baselines"]
    dss = [d for d in cfg["datasets"] if d in per_query["bm25"]]
    rows = []
    for name, by_ds in per_query.items():
        if not all(d in by_ds for d in dss):
            continue
        r = {"system": name}
        for d in dss:
            r[d] = by_ds[d].mean().item()
        r["macro"] = sum(r[d] for d in dss) / len(dss)
        rows.append(r)
    table = pl.DataFrame(rows).sort("macro", descending=True)
    tests = []
    for name, by_ds in per_query.items():
        if name not in against or not all(d in by_ds for d in dss):
            continue
        diffs = [per_query[primary][d] - by_ds[d] for d in dss]
        pt, lo, hi = paired_bootstrap(diffs, cfg["n_boot"])
        r = dict(vs=name, delta_macro=pt, ci_lo=lo, ci_hi=hi)
        for d, x in zip(dss, diffs):
            p1, l1, h1 = paired_bootstrap([x], cfg["n_boot"])
            r[f"{d}_delta"] = p1
            r[f"{d}_lo"] = l1
            r[f"{d}_hi"] = h1
        tests.append(r)
    return table, pl.DataFrame(tests).sort("delta_macro")


@app.function(hide_code=True)
def phase2b_lodo(cfg, per_query, family="indexer", kind="cal"):
    """Leave-one-dataset-out choice of the indexer layer and temperature: for each dataset, pick the
    (layer, tau) with the best mean nDCG@10 over the other datasets, then score it on the held-out one.
    Returns (per_query tensors {dataset: tensor} of the LODO teacher, the picks)."""
    dss = [d for d in cfg["datasets"] if d in per_query["bm25"]]
    cands = [n for n in per_query if n.startswith(f"{family}_L") and "_tau" in n and n.endswith(kind)]
    out, picks = {}, {}
    for d in dss:
        others = [o for o in dss if o != d]
        best = max(cands, key=lambda n: sum(per_query[n][o].mean().item() for o in others) / len(others))
        out[d] = per_query[best][d]
        picks[d] = best
    return out, picks


@app.function(hide_code=True)
def start_phase2b_first_stage(cfg, script_dir, out_dir="/root/models/phase2b", tokenizer_dir="/root/models/qwen38-nvfp4",
                              java_home="/usr/lib/jvm/java-21-openjdk-amd64"):
    """Run phase2b_bm25.py (BM25 candidates) and then phase2b_baselines.py (baseline runs) in a separate
    process. Pyserini starts a JVM through pyjnius, so it stays out of the notebook kernel. Log:
    <out_dir>/first_stage.log. Needs about 6 GB of free GPU memory for the baselines."""
    import shlex, subprocess, sys
    os.makedirs(out_dir, exist_ok=True)
    py = shlex.quote(sys.executable)
    ds = " ".join(cfg["datasets"])
    cmd = (f"{py} {shlex.quote(os.path.join(script_dir, 'phase2b_bm25.py'))} --out {out_dir} --tokenizer {tokenizer_dir} "
           f"--k {cfg['k']} --max_doc_tokens {cfg['max_doc_tokens']} --datasets {ds} && "
           f"{py} {shlex.quote(os.path.join(script_dir, 'phase2b_baselines.py'))} --dir {out_dir} --datasets {ds}")
    env = dict(os.environ, JAVA_HOME=java_home)
    return subprocess.Popen(cmd, shell=True, env=env, stdout=open(os.path.join(out_dir, "first_stage.log"), "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)


@app.cell(hide_code=True)
def _():
    stage1_button = mo.ui.run_button(label="Build BM25 candidates and baseline runs (separate process)")
    stage1_button

    return (stage1_button,)


@app.cell(hide_code=True)
def _(PHASE2B, stage1_button):
    mo.stop(not stage1_button.value, mo.md("Press the button to build the candidates. `phase2b_bm25.py` and `phase2b_baselines.py` must be in `PHASE2B['cand_dir']` (they are next to this notebook in the repository)."))
    stage1_proc = start_phase2b_first_stage(PHASE2B, PHASE2B["cand_dir"])
    mo.md(f"First stage started (pid {stage1_proc.pid}); log: `{PHASE2B['cand_dir']}/first_stage.log`.")

    return


@app.cell(hide_code=True)
def _():
    rerank_tier = mo.ui.dropdown(options=["smoke", "dev", "full"], value="dev", label="Tier")
    rerank_button = mo.ui.run_button(label="Run Phase 2b teacher job (background thread)")
    mo.hstack([rerank_tier, rerank_button], justify="start")

    return rerank_button, rerank_tier


@app.cell(hide_code=True)
def _(
    PHASE2B,
    model,
    rerank_button,
    rerank_tier,
    start_rerank_job,
    tokenizer,
    use_fast_path,
):
    mo.stop(not rerank_button.value, mo.md("Choose a tier and press **Run Phase 2b teacher job**. smoke: DL19 + DL20 (97 queries, about 3 min). "
                                          "dev: + TREC-COVID and 100 fixed random queries of NFCorpus, SciFact and FiQA (447 queries, about 20 min). "
                                          "full: all 1,418 queries. The job saves `teacher_qsa_<dataset>.pt` every 25 queries and skips queries it already has."))
    use_fast_path(True)
    _tier = PHASE2B["tiers"][rerank_tier.value]
    rerank_job = start_rerank_job(model, tokenizer, _tier["datasets"], PHASE2B["cand_dir"], PHASE2B["temps"],
                                  last_layer=PHASE2B["last_layer"], max_queries=_tier["max_queries"])
    mo.md(f"Phase 2b teacher job started (tier **{rerank_tier.value}**). Use **Refresh** below to see the progress and the results so far.")

    return


@app.cell(hide_code=True)
def _():
    p2b_refresh = mo.ui.run_button(label="Refresh Phase 2b results")
    p2b_refresh

    return (p2b_refresh,)


@app.cell(hide_code=True)
def _(PHASE2B, p2b_refresh):
    p2b_refresh
    _job = globals().get("rerank_job")
    p2b_recs = {}
    for _d in PHASE2B["datasets"]:
        _p = os.path.join(PHASE2B["cand_dir"], f"teacher_qsa_{_d}.pt")
        if os.path.exists(_p):
            p2b_recs[_d] = torch.load(_p, weights_only=False)
    mo.stop(not p2b_recs, mo.md("No Phase 2b teacher records yet."))
    p2b_per_query, p2b_qids = phase2b_evaluate(PHASE2B, p2b_recs)
    _lodo, p2b_lodo_picks = phase2b_lodo(PHASE2B, p2b_per_query)
    p2b_per_query["indexer_lodo_layer_tau_cal"] = _lodo
    p2b_table, p2b_tests = phase2b_report(PHASE2B, p2b_per_query,
                                          against=["bm25", "qrhead16_cal", "icr288_cal", "random", "oracle", "indexer_lodo_layer_tau_cal"] + PHASE2B["baselines"])
    _main = ["oracle", "ce-minilm", "icr288_cal", "qrhead16_cal",
             "indexer_L31_t1_cal", "indexer_lodo_layer_tau_cal", "indexer_zmean_t1_cal", "indexer_L31_t1_raw",
             "qrhead16_raw", "icr288_raw", "qrhead16_sparse_cal", "icr288_sparse_cal", "bm25", "random"]
    _n = {d: len(q) for d, q in p2b_qids.items()}
    mo.vstack([
        mo.md(f"Teacher job: **{_job['status'] if _job else 'not started in this session'}** {_job['progress'] if _job else ''}. "
              f"Queries with teacher records: {_n}"),
        mo.md("**nDCG@10, main systems** (only the queries with teacher records)"),
        mo.ui.table(p2b_table.filter(pl.col("system").is_in(_main)).with_columns(pl.exclude("system").round(4)), selection=None, page_size=20),
        mo.md("**Primary teacher (indexer, layer 31, tau = 1, calibrated) minus each system**: macro mean and 95% paired bootstrap interval"),
        mo.ui.table(p2b_tests.with_columns(pl.exclude("vs").round(4)), selection=None, page_size=20),
        mo.md(f"Leave-one-dataset-out picks: {p2b_lodo_picks}"),
    ])

    return p2b_per_query, p2b_recs


@app.cell(hide_code=True)
def _(PHASE2B, p2b_per_query):
    _rows = []
    for _l in PHASE2B["qsa_layers"]:
        for _t in PHASE2B["temps"]:
            for _kind in ("cal", "raw"):
                _name = f"indexer_L{_l}_tau{_t:.3f}_{_kind}"
                if _name in p2b_per_query:
                    _rows.append(dict(layer=_l, tau=round(_t, 3), kind=_kind,
                                      macro=sum(v.mean().item() for v in p2b_per_query[_name].values()) / len(p2b_per_query[_name])))
    _df = pl.DataFrame(_rows)
    mo.vstack([
        mo.md("**Indexer, every layer and temperature: macro nDCG@10 (exploratory)**"),
        alt.Chart(_df).mark_line(point=True).encode(
            x=alt.X("layer:O"), y=alt.Y("macro:Q", scale=alt.Scale(zero=False)), color="tau:N", strokeDash="kind:N"
        ).properties(width=620, height=300),
    ])

    return


@app.function(hide_code=True)
def phase2b_bias_report(cfg, teacher_recs, min_docs=10):
    """Exploratory bias check. For every query with at least min_docs candidates, the Spearman correlation of
    the teacher score with the document slot (position in the prompt) and with the document length in
    tokens, over the non-relevant candidates only (grade <= 0 or not judged). Slot is random, so any slot
    correlation is pure position bias. Returns the mean correlation per dataset and teacher."""
    n_layers = min(r["q"]["idx"].shape[0] for recs in teacher_recs.values() for r in recs.values() if r["q"] is not None)
    teachers = phase2b_teachers(cfg, n_layers)
    names = [n for n in ["indexer_L31_t1_raw", "indexer_L31_t1_cal", "qrhead16_raw", "qrhead16_cal", "icr288_raw", "icr288_cal"] if n in teachers]

    def spearman(a, b):
        ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
        ra, rb = ra - ra.mean(), rb - rb.mean()
        return (ra @ rb / (ra.norm() * rb.norm()).clamp_min(1e-12)).item()

    rows = []
    for ds, recs in teacher_recs.items():
        data = torch.load(os.path.join(cfg["cand_dir"], f"candidates_{ds}.pt"), weights_only=False)
        for n in names:
            rho_slot, rho_len = [], []
            for q, r in recs.items():
                if len(r["docids"]) < min_docs:
                    continue
                keep = torch.tensor([data["qrels"][q].get(d, 0) <= 0 for d in r["docids"]])
                if int(keep.sum()) < min_docs:
                    continue
                sc = teachers[n](r["q"], r["null"])[keep]
                slot = torch.arange(len(r["docids"]), dtype=torch.float32)[keep]
                ln = torch.tensor([len(data["docs"][d]["ids"]) for d in r["docids"]], dtype=torch.float32)[keep]
                rho_slot.append(spearman(sc, slot))
                rho_len.append(spearman(sc, ln))
            rows.append(dict(dataset=ds, teacher=n, queries=len(rho_slot),
                             rho_slot=sum(rho_slot) / max(1, len(rho_slot)), rho_len=sum(rho_len) / max(1, len(rho_len))))
    return pl.DataFrame(rows)


@app.cell(hide_code=True)
def _(PHASE2B, p2b_recs):
    p2b_bias = phase2b_bias_report(PHASE2B, p2b_recs)
    mo.vstack([
        mo.md("**Bias check (exploratory)**: mean per-query Spearman correlation of the score with the document slot and with the "
              "document length, on non-relevant candidates. The slot is random, so a slot correlation is position bias."),
        mo.ui.table(p2b_bias.with_columns(pl.col("rho_slot", "rho_len").round(3)), selection=None, page_size=40),
    ])

    return


if __name__ == "__main__":
    app.run()
