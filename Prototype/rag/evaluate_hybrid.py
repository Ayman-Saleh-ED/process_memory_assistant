"""Compare keyword, semantic and hybrid manual search on the same test questions.

Run from the Prototype folder:
    pip install fastembed
    python rag/hybrid.py             # once: builds rag/page_embeddings.npz
    python rag/evaluate_hybrid.py
Saves results/rag_retrieval_comparison.csv
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "rag")
from hybrid import HybridRetriever
from questions import IN_SCOPE, OUT_OF_SCOPE


def evaluate(retriever, modes=("keyword", "semantic", "hybrid")):
    chunks = json.loads(Path("rag/chunks.json").read_text(encoding="utf-8"))
    pages = {}
    for c in chunks:
        pages[(c["doc"], c["page"])] = pages.get((c["doc"], c["page"]), "") + " " + c["text"]
    rows = []
    for mode in modes:
        hits = {1: 0, 3: 0, 5: 0}
        for q, doc, rx in IN_SCOPE:
            gold = {k for k, t in pages.items() if (doc is None or k[0] == doc) and re.search(rx, t, re.I)}
            found = retriever.search(q, k=5, mode=mode)
            for k in hits:
                if any((f["doc"], f["page"]) in gold for f in found[:k]):
                    hits[k] += 1
        n = len(IN_SCOPE)
        rows.append({"search": mode, **{f"top{k}": round(hits[k] / n, 3) for k in hits}, "questions": n})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = evaluate(HybridRetriever("rag/chunks.json", "rag/page_embeddings.npz"))
    print(table.to_string(index=False))
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/rag_retrieval_comparison.csv", index=False)
