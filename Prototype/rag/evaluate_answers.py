"""Evaluate the WRITTEN ANSWERS, not only the search: do the quotes check out, and does it refuse when the manuals do not answer?

Needs a Groq key (environment variable GROQ_API_KEY, or .streamlit/secrets.toml) and internet.
Run from the Prototype folder:
    python rag/evaluate_answers.py             # uses keyword search (what is deployed now)
    python rag/evaluate_answers.py hybrid      # uses the hybrid search (needs rag/page_embeddings.npz)
About 2-3 minutes. Saves results/rag_answer_eval.csv
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "rag")
from llm_answer import answer
from questions import IN_SCOPE, OUT_OF_SCOPE
from retrieval import Retriever


def get_key():
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    secrets = Path(".streamlit/secrets.toml")
    if secrets.exists():
        m = re.search(r'GROQ_API_KEY\s*=\s*"([^"]+)"', secrets.read_text())
        if m:
            return m.group(1)
    sys.exit("No Groq key found")


def main(mode):
    key = get_key()
    if mode == "hybrid":
        from hybrid import HybridRetriever
        retriever = HybridRetriever("rag/chunks.json", "rag/page_embeddings.npz")
    else:
        retriever = Retriever("rag/chunks.json")
    chunks = json.loads(Path("rag/chunks.json").read_text(encoding="utf-8"))
    pages = {}
    for c in chunks:
        pages[(c["doc"], c["page"])] = pages.get((c["doc"], c["page"]), "") + " " + c["text"]

    rows, failed = [], []
    questions = [(q, doc, rx, "in scope") for q, doc, rx in IN_SCOPE] + [(q, None, None, "out of scope") for q in OUT_OF_SCOPE]
    for q, doc, rx, kind in questions:
        sources = retriever.search(q, k=5)
        try:
            res = answer(key, "Question: " + q, sources)
        except Exception as e:
            print("FAILED:", q, "|", str(e)[:300])
            failed.append(q)
            continue
        on_gold = None
        if rx and res["points"]:
            gold = {k for k, t in pages.items() if (doc is None or k[0] == doc) and re.search(rx, t, re.I)}
            on_gold = sum((sources[p["source"] - 1]["doc"], sources[p["source"] - 1]["page"]) in gold for p in res["points"]) / len(res["points"])
        rows.append({"kind": kind, "question": q, "points_kept": len(res["points"]), "points_dropped": res["dropped"],
                     "share_on_a_correct_page": on_gold, "model": res["model"]})
        print(f"{kind:13} kept {len(res['points'])} dropped {res['dropped']}  ({res['model']})  {q}", flush=True)
        time.sleep(20)      # the free Groq plan limits tokens per minute; each request is a few thousand tokens

    df = pd.DataFrame(rows)
    inn, out = df[df.kind == "in scope"], df[df.kind == "out of scope"]
    kept, dropped = inn.points_kept.sum(), inn.points_dropped.sum()
    print("\n--- summary (search: %s) ---" % mode)
    if failed:
        print(f"WARNING: {len(failed)} of {len(questions)} questions FAILED to get any reply, so these numbers are partial. Wait a few minutes and run again.")
    print(f"In-scope questions: {len(inn)}")
    print(f"  got an answer (at least 1 verified point): {(inn.points_kept > 0).mean():.0%}")
    print(f"  points that passed the quote check: {kept}/{kept + dropped} = {kept / max(kept + dropped, 1):.0%}")
    print(f"  verified points that cite a correct page: {inn.share_on_a_correct_page.mean():.0%} (average over questions with an answer)")
    print(f"Out-of-scope questions: {len(out)}")
    print(f"  correctly gave no answer: {(out.points_kept == 0).mean():.0%}")
    df.to_csv("results/rag_answer_eval.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "keyword")
