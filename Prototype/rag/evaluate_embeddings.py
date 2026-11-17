"""OPTIONAL, run on your own PC (needs internet the first time to download the model).
Compares embedding search with the keyword search on the same test questions.

    pip install sentence-transformers
    python rag/evaluate_embeddings.py
"""
import json, re, sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "rag")
from questions import IN_SCOPE, OUT_OF_SCOPE

chunks = json.loads(Path("rag/chunks.json").read_text(encoding="utf-8"))
pages = {}
for c in chunks:
    pages.setdefault((c["doc"], c["page"]), "")
    pages[(c["doc"], c["page"])] += " " + c["text"]

model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)


def search(question, k=5):
    q = model.encode([question], normalize_embeddings=True)[0]
    scores = chunk_vectors @ q
    results, seen = [], set()
    for i in scores.argsort()[::-1]:
        key = (chunks[i]["doc"], chunks[i]["page"])
        if key in seen:
            continue
        seen.add(key)
        results.append((key, float(scores[i])))
        if len(results) == k:
            break
    return results


hits = {1: 0, 3: 0, 5: 0}
top_scores = []
for q, doc, rx in IN_SCOPE:
    gold = {k for k, t in pages.items() if (doc is None or k[0] == doc) and re.search(rx, t, re.I)}
    found = search(q)
    top_scores.append(found[0][1])
    for k in hits:
        if any(f[0] in gold for f in found[:k]):
            hits[k] += 1
n = len(IN_SCOPE)
for k in hits:
    print(f"Right page in top {k}: {hits[k]}/{n} = {hits[k]/n:.0%}")
print("Top score, in-scope: min %.2f, median %.2f" % (min(top_scores), np.median(top_scores)))
print("Top score, out-of-scope:", [round(search(q, 1)[0][1], 2) for q in OUT_OF_SCOPE])
