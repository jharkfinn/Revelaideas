"""First stage for the OBLIQ-Bench Congress comparison: Qwen3-Embedding retrieval over the full corpus.

    python obliq_embed.py --base /root/models/obliq/tip-of-tongue/congress --out /root/models/obliq \
        --models Qwen/Qwen3-Embedding-0.6B Qwen/Qwen3-Embedding-4B --max_len 512 --topk 1000

For each model: encodes every passage (no instruction) and every query (with the Qwen3-Embedding retrieval
instruction), last-token pooling with left padding, L2-normalized, cosine similarity. Writes
<out>/run_<short>.pt = {qid: [(docid, score), ...top-k]} and prints NDCG@10 / Recall@10/50/100 against the
gold qrels, to compare with Table 4 of the paper (Qwen3-Embed-0.6B: NDCG@10 .006, R@100 .055;
Qwen3-Embed-4B: NDCG@10 .040, R@100 .122).
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"


@torch.no_grad()
def encode(model, tok, texts, max_len, batch_tokens=65536, device="cuda"):
    """Last-token embeddings (left padding), L2-normalized, fp16 on the CPU, in the input order."""
    # sort by character length (a proxy for token length) so each batch pads little; no upfront tokenization
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out = torch.empty(len(texts), model.config.hidden_size, dtype=torch.float16)
    i = 0
    while i < len(order):
        L = min(max_len, len(texts[order[i]]) // 3 + 8)
        n = max(1, batch_tokens // max(L, 1))
        idx = order[i : i + n]
        enc = tok([texts[k] for k in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state[:, -1]
        out[torch.tensor(idx)] = F.normalize(h.float(), dim=-1).half().cpu()
        i += n
    return out


def metrics(run, qrels):
    """Mean NDCG@10 and Recall@k (one or more gold docs per query, binary)."""
    res = {"ndcg@10": 0.0, "r@10": 0.0, "r@50": 0.0, "r@100": 0.0}
    for q, gold in qrels.items():
        ranked = [d for d, _ in run.get(q, [])]
        dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in gold)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
        res["ndcg@10"] += dcg / idcg
        for k in (10, 50, 100):
            res[f"r@{k}"] += len(set(ranked[:k]) & gold) / len(gold)
    return {k: round(v / len(qrels), 4) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B"])
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--topk", type=int, default=1000)
    args = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer
    queries = [json.loads(l) for l in open(f"{args.base}/queries+qrels/queries.jsonl")]
    qrels = {}
    for line in open(f"{args.base}/queries+qrels/qrels.tsv"):
        q, d, s = line.rstrip("\n").split("\t")
        if q != "query-id" and int(s) > 0:
            qrels.setdefault(q, set()).add(d)
    ids, texts = [], []
    for line in open(f"{args.base}/corpus/corpus.jsonl"):
        r = json.loads(line)
        ids.append(r["_id"])
        texts.append(" ".join(r["text"].split()))
    for name in args.models:
        short = name.split("/")[-1]
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        model = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        emb_path = os.path.join(args.out, f"emb_{short}.pt")
        if os.path.exists(emb_path):
            D = torch.load(emb_path)
        else:
            D = encode(model, tok, texts, args.max_len)
            torch.save(D, emb_path)
        t_doc = time.time() - t0
        Q = encode(model, tok, [f"Instruct: {INSTRUCT}\nQuery:{q['text']}" for q in queries], args.max_len)
        Dg = D.cuda()
        run = {}
        for b in range(0, len(queries), 64):
            s = Q[b : b + 64].cuda().float() @ Dg.float().T
            top = s.topk(args.topk, dim=-1)
            for j, q in enumerate(queries[b : b + 64]):
                run[q["_id"]] = [(ids[i], float(v)) for v, i in zip(top.values[j].tolist(), top.indices[j].tolist())]
        torch.save(run, os.path.join(args.out, f"run_{short}.pt"))
        print(json.dumps(dict(model=short, docs=len(ids), seconds_docs=round(t_doc), seconds_total=round(time.time() - t0),
                              **metrics(run, qrels))), flush=True)
        del model, Dg
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
