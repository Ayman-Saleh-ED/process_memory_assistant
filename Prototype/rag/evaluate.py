"""Run from the Prototype folder:  python rag/evaluate.py
Gold pages = pages whose text matches the regex in rag/questions.py (independent of the search).
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "rag")
from retrieval import Retriever
from questions import IN_SCOPE, OUT_OF_SCOPE
import numpy as np

chunks = json.loads(Path("rag/chunks.json").read_text(encoding="utf-8"))
pages = {}
for c in chunks:
    pages.setdefault((c["doc"], c["page"]), "")
    pages[(c["doc"], c["page"])] += " " + c["text"]

r = Retriever("rag/chunks.json")
hits = {1: 0, 3: 0, 5: 0}
top_scores = []
for q, doc, rx in IN_SCOPE:
    gold = {k for k, t in pages.items() if (doc is None or k[0] == doc) and re.search(rx, t, re.I)}
    found = r.search(q, k=5)
    top_scores.append(found[0]["score"])
    for k in hits:
        if any((f["doc"], f["page"]) in gold for f in found[:k]):
            hits[k] += 1
n = len(IN_SCOPE)
print("Questions:", n)
for k in hits:
    print(f"Right page in top {k}: {hits[k]}/{n} = {hits[k]/n:.0%}")
print("Top score, in-scope questions: min %.2f, median %.2f" % (min(top_scores), np.median(top_scores)))
out = [r.search(q, k=1)[0]["score"] for q in OUT_OF_SCOPE]
print("Top score, out-of-scope questions:", [round(x, 2) for x in out])
