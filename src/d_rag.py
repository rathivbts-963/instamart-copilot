"""Part D tasks 1-4 -- chunk the policies, TF-IDF retrieval, grounded answering, evaluation."""
import glob
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mock_llm  # noqa: E402

POLICY_DIR = "data/policies"
CHUNK_WORDS = 40

SYSTEM = ("You are the Instamart POLICY ASSISTANT. Answer only from the supplied CONTEXT. "
          "If the answer is not in the context, reply: I could not find that information in the policy documents.")


def chunk_documents(size=CHUNK_WORDS, overlap=10):
    chunks = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        start, step = 0, size - overlap
        while True:
            chunks.append({"source": os.path.basename(path), "text": " ".join(words[start:start + size])})
            if start + size >= len(words):          # this chunk reaches the last word
                break
            start += step
    return chunks


class Retriever:
    def __init__(self, overlap=10):
        self.chunks = chunk_documents(overlap=overlap)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        scores = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(scores[i]), 4),
                 "text": self.chunks[i]["text"]} for i in top]


_default = None


def retrieve(query, k=3):
    global _default
    if _default is None:
        _default = Retriever()
    return _default.retrieve(query, k)


def answer(question, retriever=None):
    hits = retriever.retrieve(question, 3) if retriever else retrieve(question, 3)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return {"answer": resp.choices[0].message.content, "sources": [h["source"] for h in hits]}


def evaluate(retriever):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    rows = []
    for q in ev.itertuples():
        res = answer(q.question, retriever)
        rows.append({"question": q.question, "expected_doc": q.expected_doc,
                     "doc_in_top3": q.expected_doc in res["sources"],
                     "fact_in_answer": bool(q.expected_fact) and q.expected_fact in res["answer"],
                     "fallback": res["answer"].startswith("I could not find"), "answer": res["answer"]})
    df = pd.DataFrame(rows)
    ans, no = df[df["expected_doc"] != "NONE"], df[df["expected_doc"] == "NONE"]
    summary = {"doc_in_top3": f"{ans['doc_in_top3'].sum()}/{len(ans)}",
               "fact_in_answer": f"{ans['fact_in_answer'].sum()}/{len(ans)}",
               "fallback_on_unanswerable": f"{no['fallback'].sum()}/{len(no)}"}
    return df, summary


def main():
    for ov in (10, 0):
        r = Retriever(overlap=ov)
        counts = pd.Series([c["source"] for c in r.chunks]).value_counts().to_dict()
        print(f"\n=== overlap {ov}: {len(r.chunks)} chunks {counts}")
        if ov == 10:
            for h in r.retrieve("refund policy late delivery"):
                print(f"  top3 for 'refund policy late delivery': {h['source']} score {h['score']}")
        df, summary = evaluate(r)
        print(f"  evaluation: {summary}")
        bad = df[(df["expected_doc"] != "NONE") & ~(df["doc_in_top3"] & df["fact_in_answer"])]
        for b in bad.itertuples():
            print(f"  FAILED: {b.question!r} doc_in_top3={b.doc_in_top3} answer={b.answer!r}")


if __name__ == "__main__":
    main()
