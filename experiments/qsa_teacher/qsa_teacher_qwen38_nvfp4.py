# /// script
# requires-python = ">=3.13"
# dependencies = [
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


if __name__ == "__main__":
    app.run()

