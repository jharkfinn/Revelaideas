"""Phase 2b baselines: rerank the BM25 top-k candidates with standard models (GPU, own process).

    python phase2b_baselines.py --dir /root/models/phase2b --datasets scifact nfcorpus ... --models ce-minilm ...

Reads <dir>/candidates_<name>.pt (phase2b_bm25.py) and writes <dir>/run_<model>_<name>.pt =
{qid: {docid: score}}. Every model reads the same truncated document text (text_trunc).

Models:
  ce-minilm    cross-encoder/ms-marco-MiniLM-L-6-v2 (supervised on MS MARCO; BEIR paper reranker)
  bge-m3-rr    BAAI/bge-reranker-v2-m3 (strong supervised cross-encoder, 568M)
  contriever   facebook/contriever (unsupervised dense, mean pooling, dot product)
  revela-500m  trumancai/Revela-500M LoRA on Qwen/Qwen2.5-0.5B (this repo: "query: " / "passage: " prefixes,
               EOS appended, EOS pooling, L2-normalized, cosine)
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F


def batches(items, n):
    for i in range(0, len(items), n):
        yield items[i : i + n]


@torch.no_grad()
def cross_encoder_scores(name, pairs, max_len=512, bs=64, dtype=torch.float16):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=dtype).cuda().eval()
    out = []
    for b in batches(pairs, bs):
        enc = tok([q for q, _ in b], [d for _, d in b], padding=True, truncation="only_second",
                  max_length=max_len, return_tensors="pt").to("cuda")
        out += model(**enc).logits[:, 0].float().tolist()
    del model
    return out


@torch.no_grad()
def contriever_embed(model, tok, texts, max_len=512, bs=128):
    out = []
    for b in batches(texts, bs):
        enc = tok(b, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
        h = model(**enc).last_hidden_state.float()
        m = enc["attention_mask"][..., None].float()
        out.append((h * m).sum(1) / m.sum(1))
    return torch.cat(out)


@torch.no_grad()
def revela_embed(model, tok, texts, max_len=512, bs=64):
    out = []
    for b in batches(texts, bs):
        ids = tok(b, add_special_tokens=True, truncation=True, max_length=max_len - 1).input_ids
        ids = [x + [tok.eos_token_id] for x in ids]
        enc = tok.pad({"input_ids": ids}, padding=True, return_attention_mask=True, return_tensors="pt").to("cuda")
        h = model(**enc).last_hidden_state
        last = enc["attention_mask"].sum(1) - 1
        out.append(F.normalize(h[torch.arange(h.shape[0], device=h.device), last].float(), dim=-1))
    return torch.cat(out)


def dense_scores(kind, data):
    from transformers import AutoModel, AutoTokenizer
    if kind == "contriever":
        tok = AutoTokenizer.from_pretrained("facebook/contriever")
        model = AutoModel.from_pretrained("facebook/contriever", torch_dtype=torch.float16).cuda().eval()
        qfmt, dfmt = (lambda q: q), (lambda d: d)
        embed = contriever_embed
    else:
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        tok.padding_side = "right"
        if tok.pad_token_id is None:
            tok.pad_token_id = 0
        base = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, "trumancai/Revela-500M").merge_and_unload().cuda().eval()
        qfmt = lambda q: f"query: {q.strip()}"
        dfmt = lambda d: f"passage: {d.replace(chr(10), ' ', 1).strip()}"
        embed = revela_embed
    qids = sorted(data["queries"])
    docids = sorted(data["docs"])
    q_emb = embed(model, tok, [qfmt(data["queries"][q]) for q in qids])
    d_emb = embed(model, tok, [dfmt(data["docs"][d]["text_trunc"]) for d in docids])
    qi = {q: i for i, q in enumerate(qids)}
    di = {d: i for i, d in enumerate(docids)}
    run = {}
    for q in qids:
        cand = [d for d, _ in data["cands"][q]]
        s = d_emb[[di[d] for d in cand]] @ q_emb[qi[q]]
        run[q] = dict(zip(cand, s.tolist()))
    del model
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["ce-minilm", "bge-m3-rr", "contriever", "revela-500m"])
    args = ap.parse_args()
    ce = {"ce-minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2", "bge-m3-rr": "BAAI/bge-reranker-v2-m3"}
    for name in args.datasets:
        data = torch.load(os.path.join(args.dir, f"candidates_{name}.pt"), weights_only=False)
        for m in args.models:
            path = os.path.join(args.dir, f"run_{m}_{name}.pt")
            if os.path.exists(path):
                continue
            t0 = time.time()
            if m in ce:
                keys = [(q, d) for q in sorted(data["queries"]) for d, _ in data["cands"][q]]
                pairs = [(data["queries"][q], data["docs"][d]["text_trunc"]) for q, d in keys]
                scores = cross_encoder_scores(ce[m], pairs)
                run = {}
                for (q, d), s in zip(keys, scores):
                    run.setdefault(q, {})[d] = s
            else:
                run = dense_scores(m, data)
            torch.save(run, path)
            print(f"{name} {m}: {len(run)} queries, {time.time() - t0:.0f} s", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
