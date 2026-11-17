import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    """Finds the manual pages that best match a question (keyword search, TF-IDF)."""

    def __init__(self, chunks_path):
        self.chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                          sublinear_tf=True, min_df=2)
        self.matrix = self.vectorizer.fit_transform([(c["section"] + ". " if c["section"] else "") + c["text"] for c in self.chunks])

    def search(self, question, k=3):
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        results, seen = [], set()
        for i in scores.argsort()[::-1]:
            c = self.chunks[i]
            if (c["doc"], c["page"]) in seen:   # one result per page
                continue
            seen.add((c["doc"], c["page"]))
            results.append({"title": c["title"], "section": c["section"], "doc": c["doc"], "page": c["page"],
                            "text": c["text"], "score": float(scores[i])})
            if len(results) == k:
                break
        return results
