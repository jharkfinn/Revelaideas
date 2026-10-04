"""Phase 2b first stage: BM25 top-k candidates from Pyserini prebuilt Lucene indexes.

Run in its own process with JAVA_HOME set (pyjnius starts a JVM, keep it out of the notebook kernel):

    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 python phase2b_bm25.py --out /root/models/phase2b \
        --tokenizer /root/models/qwen38-nvfp4 --datasets dl19 dl20 trec-covid nfcorpus scifact fiqa

For each dataset it writes <out>/candidates_<name>.pt with
    queries {qid: text}            only queries that have qrels
    qrels   {qid: {docid: grade}}  as given by Pyserini (BEIR test qrels, TREC DL passage qrels)
    cands   {qid: [(docid, bm25_score), ...]}  BM25 order, k hits
    docs    {docid: dict(text=full document string, ids=first max_doc_tokens Qwen token ids,
                         text_trunc=decode(ids))}
The document string is "title\\ntext" (BEIR, when the title is not empty) or the passage (MS MARCO).
Every reranker in Phase 2b reads the same truncated document (text_trunc or ids).
"""
import argparse
import json
import os
import time

import torch
from pyserini.search import get_qrels, get_topics
from pyserini.search.lucene import LuceneSearcher

# name -> (prebuilt index, topics, qrels). BM25 defaults k1 = 0.9, b = 0.4 (Pyserini regressions).
DATASETS = {
    "dl19": ("msmarco-v1-passage", "dl19-passage", "dl19-passage"),
    "dl20": ("msmarco-v1-passage", "dl20", "dl20-passage"),
    "trec-covid": ("beir-v1.0.0-trec-covid.flat", "beir-v1.0.0-trec-covid-test", "beir-v1.0.0-trec-covid-test"),
    "nfcorpus": ("beir-v1.0.0-nfcorpus.flat", "beir-v1.0.0-nfcorpus-test", "beir-v1.0.0-nfcorpus-test"),
    "scifact": ("beir-v1.0.0-scifact.flat", "beir-v1.0.0-scifact-test", "beir-v1.0.0-scifact-test"),
    "fiqa": ("beir-v1.0.0-fiqa.flat", "beir-v1.0.0-fiqa-test", "beir-v1.0.0-fiqa-test"),
}


def doc_string(raw):
    d = json.loads(raw)
    if "contents" in d:
        return d["contents"].strip()
    title, text = d.get("title", "").strip(), d.get("text", "").strip()
    return f"{title}\n{text}" if title else text


def build(name, k, tokenizer, max_doc_tokens, searchers):
    index, topics_name, qrels_name = DATASETS[name]
    if index not in searchers:
        searchers[index] = LuceneSearcher.from_prebuilt_index(index)
    searcher = searchers[index]
    topics = get_topics(topics_name)
    qrels = {str(q): {str(d): int(g) for d, g in v.items()} for q, v in get_qrels(qrels_name).items()}
    queries = {str(q): t["title"].strip() for q, t in topics.items() if str(q) in qrels}
    qids = sorted(queries)
    hits = searcher.batch_search([queries[q] for q in qids], qids, k=k, threads=8)
    cands = {q: [(h.docid, float(h.score)) for h in hits[q]] for q in qids}
    docs = {}
    for q in qids:
        for docid, _ in cands[q]:
            if docid not in docs:
                text = doc_string(searcher.doc(docid).raw())
                ids = tokenizer(text, add_special_tokens=False).input_ids[:max_doc_tokens]
                docs[docid] = dict(text=text, ids=ids, text_trunc=tokenizer.decode(ids))
    return dict(name=name, index=index, k=k, max_doc_tokens=max_doc_tokens,
                queries=queries, qrels={q: qrels[q] for q in qids}, cands=cands, docs=docs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--max_doc_tokens", type=int, default=256)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out, exist_ok=True)
    searchers = {}
    for name in args.datasets:
        t0 = time.time()
        res = build(name, args.k, tokenizer, args.max_doc_tokens, searchers)
        torch.save(res, os.path.join(args.out, f"candidates_{name}.pt"))
        n_tok = sum(len(d["ids"]) for d in res["docs"].values()) / max(1, len(res["docs"]))
        print(json.dumps(dict(dataset=name, queries=len(res["queries"]), docs=len(res["docs"]),
                              mean_doc_tokens=round(n_tok, 1), seconds=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
