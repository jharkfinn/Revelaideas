"""First stage for the OBLIQ-Bench Congress pool with a learned sparse encoder (SPLADE-style), e.g.
Linkup-Platform/linkup-sparseup-embed-v1 (SPARSEUP, ModernBERT 149M, from LateOn-unsupervised).

    python obliq_sparse.py --base /root/models/obliq/tip-of-tongue/congress --out /root/models/obliq \
        --model Linkup-Platform/linkup-sparseup-embed-v1 --doc_max_len 512 --query_max_len 256 --topk 1000

Uses the model's own tokenization, "[Q] "/"[D] " prefixes (attended, not pooled), activation, per-position
top-k, max-pooling and vocab folding (the model's forward), but keeps only the nonzero terms of each passage,
because dense float32 rows over the ~50K vocabulary would take ~43 GB for 213,650 passages. Scores are dot
products, computed for all queries with one sparse-dense matrix product on the GPU. Writes
<out>/run_<short>.pt = {qid: [(docid, score), ...top-k]} and prints NDCG@10 / Recall@10/50/100.
"""
import argparse
import json
import math
import os
import time

import torch


@torch.inference_mode()
def encode_sparse(model, texts, kind, max_len, batch_size, device="cuda"):
    """-> (crow [N + 1], col [nnz] int32, val [nnz] float32) CSR rows in the input order."""
    prefix = model.config.query_prefix if kind == "query" else model.config.document_prefix
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    rows_col, rows_val = [None] * len(texts), [None] * len(texts)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        ids, attn, pool, _ = model._tokenize([texts[k] for k in idx], prefix, max_len)
        sp = model(ids.to(device), attn.to(device), pool.to(device)).float()
        r, c = torch.nonzero(sp, as_tuple=True)  # row-major, so grouped by row
        v = sp[r, c].cpu()
        counts = torch.bincount(r, minlength=len(idx)).tolist()
        for k, cc, vv in zip(idx, torch.split(c.to(torch.int32).cpu(), counts), torch.split(v, counts)):
            rows_col[k], rows_val[k] = cc, vv
    lens = torch.tensor([c.numel() for c in rows_col])
    crow = torch.zeros(len(texts) + 1, dtype=torch.int64)
    crow[1:] = lens.cumsum(0)
    return crow, torch.cat(rows_col), torch.cat(rows_val)


def metrics(run, qrels):
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
    ap.add_argument("--model", default="Linkup-Platform/linkup-sparseup-embed-v1")
    ap.add_argument("--doc_max_len", type=int, default=512)
    ap.add_argument("--query_max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--topk", type=int, default=1000)
    args = ap.parse_args()
    from transformers import AutoModel
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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
    short = args.model.split("/")[-1]
    t0 = time.time()
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32).cuda().eval()
    V = model.config.vocab_size
    path = os.path.join(args.out, f"sparse_{short}.pt")
    if os.path.exists(path):
        crow, col, val = torch.load(path)
    else:
        crow, col, val = encode_sparse(model, texts, "document", args.doc_max_len, args.batch_size)
        torch.save((crow, col, val), path)
    t_doc = time.time() - t0
    qc, qcol, qval = encode_sparse(model, [q["text"] for q in queries], "query", args.query_max_len, args.batch_size)
    Q = torch.sparse_csr_tensor(qc, qcol.long(), qval, size=(len(queries), V)).to_dense().cuda()
    D = torch.sparse_csr_tensor(crow.cuda(), col.long().cuda(), val.cuda(), size=(len(ids), V))
    S = (D @ Q.T).T  # [Nq, N]
    top = S.topk(args.topk, dim=-1)
    run = {q["_id"]: [(ids[i], float(v)) for v, i in zip(top.values[j].tolist(), top.indices[j].tolist())]
           for j, q in enumerate(queries)}
    torch.save(run, os.path.join(args.out, f"run_{short}.pt"))
    print(json.dumps(dict(model=short, docs=len(ids), mean_doc_terms=round(col.numel() / len(ids), 1),
                          seconds_docs=round(t_doc), seconds_total=round(time.time() - t0), **metrics(run, qrels))), flush=True)


if __name__ == "__main__":
    main()
