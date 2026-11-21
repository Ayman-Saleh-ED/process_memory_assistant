"""Hybrid manual search: keyword (TF-IDF) + semantic (small embedding model), merged by rank.

The embedding model (BAAI/bge-small-en-v1.5 through the `fastembed` package, no torch needed) is only used to
embed the question. The manual pages are embedded once, offline, and saved in rag/page_embeddings.npz.

    pip install fastembed
    python rag/hybrid.py          # builds rag/page_embeddings.npz (a few minutes the first time)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval import Retriever

MODEL = "BAAI/bge-small-en-v1.5"
PIECE = 1200        # characters; long pages are cut into pieces, a page scores as its best piece
RRF_K = 60          # reciprocal rank fusion constant


class FastEmbedder:
    def __init__(self):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(MODEL)

    @staticmethod
    def _norm(v):
        v = np.asarray(list(v), dtype=np.float32)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def passages(self, texts):
        return self._norm(self.model.passage_embed(texts))

    def query(self, text):
        return self._norm(self.model.query_embed([text]))[0]


def cut(text, size=PIECE):
    words, pieces, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > size and cur:
            pieces.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        pieces.append(cur)
    return pieces or [""]


def build_embeddings(chunks_path, out_path, embedder=None):
    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    embedder = embedder or FastEmbedder()
    texts, owner = [], []
    for i, c in enumerate(chunks):
        for p in cut(c["text"]):
            texts.append((c["section"] + ". " if c["section"] else "") + p)
            owner.append(i)
    vectors = embedder.passages(texts)
    np.savez_compressed(out_path, vectors=vectors.astype(np.float16), owner=np.array(owner), n_chunks=len(chunks))
    print("saved", out_path, vectors.shape)


class HybridRetriever(Retriever):
    def __init__(self, chunks_path, embeddings_path, embedder=None):
        super().__init__(chunks_path)
        data = np.load(embeddings_path)
        if int(data["n_chunks"]) != len(self.chunks):
            raise ValueError("page_embeddings.npz does not match chunks.json: build it again")
        self.vectors = data["vectors"].astype(np.float32)
        self.owner = data["owner"]
        self._embedder = embedder

    def _keyword_scores(self, question):
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]

    def _semantic_scores(self, question):
        if self._embedder is None:
            self._embedder = FastEmbedder()
        piece_scores = self.vectors @ self._embedder.query(question)
        scores = np.full(len(self.chunks), -1.0, dtype=np.float32)
        np.maximum.at(scores, self.owner, piece_scores)
        return scores

    def search(self, question, k=3, mode="hybrid"):
        kw = self._keyword_scores(question)
        self.last_mode = mode
        sem = None
        if mode != "keyword":
            try:
                sem = self._semantic_scores(question)
            except Exception:           # package or model not available: keyword search only, the app keeps working
                mode = self.last_mode = "keyword"
        if mode == "keyword":
            scores = kw
        elif mode == "semantic":
            scores = sem
        else:   # reciprocal rank fusion: a page ranked high by both searches wins
            rank_kw = np.empty(len(kw), dtype=int); rank_kw[kw.argsort()[::-1]] = np.arange(len(kw))
            rank_sem = np.empty(len(sem), dtype=int); rank_sem[sem.argsort()[::-1]] = np.arange(len(sem))
            scores = (1 / (RRF_K + rank_kw) + 1 / (RRF_K + rank_sem)) / (2 / (RRF_K + 1))   # 1.0 = first in both
        results, seen = [], set()
        for i in scores.argsort()[::-1]:
            c = self.chunks[i]
            if (c["doc"], c["page"]) in seen:
                continue
            seen.add((c["doc"], c["page"]))
            results.append({"title": c["title"], "section": c["section"], "doc": c["doc"], "page": c["page"],
                            "text": c["text"], "score": float(scores[i])})
            if len(results) == k:
                break
        return results


if __name__ == "__main__":
    build_embeddings("rag/chunks.json", "rag/page_embeddings.npz")
