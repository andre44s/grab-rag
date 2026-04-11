import re
import sys
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import normalize, contains_answer

punct_re=re.compile(r"[^\w\s]", re.UNICODE)

models_dir = Path(__file__).parent.parent / "models"
bge_model = str(models_dir / "bge")
bge_prefix = "Represent this sentence for searching relevant passages: "

#fixed pool size for stable hybrid scores
dense_k=80

class Retriever:
    def __init__(self, passages, model_name=bge_model, alpha=0.5, batch_size=64):
        self.passages = passages
        self.alpha = alpha

        #bm25
        tokenized=[]
        for p in passages:
            txt = punct_re.sub("", p["text"])
            tokenized.append(txt.lower().split())
        self.bm25 = BM25Okapi(tokenized)

        #dense
        self.encoder = SentenceTransformer(model_name,device="cuda")
        texts=[]
        for p in passages:
            texts.append(p['text'])
        embs = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(passages) > 500,
            normalize_embeddings=True,
        )
        embs = embs.astype(np.float32)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    @classmethod
    def from_records(cls, records, **kwargs):
        seen = set()
        passages=[]
        for rec in records:
            gp = rec.get("gold_passages") or []
            for j in range(len(gp)):
                p=gp[j]
                text = p.get('text', '').strip()
                if text and text not in seen:
                    seen.add(text)
                    passages.append({
                        "pid": f"{rec.get('id','')}_{j}",
                        'title': p.get('title',''),
                        "text": text,
                    })
        return cls(passages, **kwargs)

    def retrieve(self, query, top_k=5, exclude_pids=None):
        #print("retrieve:", query[:50], "top_k:", top_k)
        n=len(self.passages)

        #bm25
        qtoks = punct_re.sub("", query)
        bm25_raw = np.array(self.bm25.get_scores(qtoks.lower().split()),dtype=np.float32)

        #dense
        q_emb = self.encoder.encode([bge_prefix + query], normalize_embeddings=True)
        q_emb = q_emb.astype(np.float32)
        dense_raw = np.zeros(n,dtype=np.float32)
        fetch_k=min(n, dense_k)
        scores_top, indices_top = self.index.search(q_emb, fetch_k)
        for idx, s in zip(indices_top[0],scores_top[0]):
            if 0 <= idx < n:
                dense_raw[idx] = s

        #hybrid
        #print("bm25 max:", bm25_raw.max(), "dense max:", dense_raw.max())
        hybrid = self.alpha * minmax(dense_raw) + (1 - self.alpha) * minmax(bm25_raw)

        ranked=np.argsort(-hybrid)
        results=[]
        for idx in ranked:
            if len(results) >= top_k:
                break
            p = self.passages[int(idx)]
            if exclude_pids and p["pid"] in exclude_pids:
                continue
            results.append({
                "pid": p["pid"],
                "title": p["title"],
                "text": p["text"],
                'score': float(hybrid[idx]),
            })
        return results

    def hard_negatives(self, query, gold_passages, answers, top_k=80, candidates=None):
        gold_norms=set()
        for p in gold_passages:
            gold_norms.add(normalize(p.get('text','')))
        if candidates is None:
            candidates = self.retrieve(query, top_k=top_k)
        results = [r for r in candidates if normalize(r["text"]) not in gold_norms]
        results = [r for r in results if not contains_answer(r.get("title","") + " " + r["text"], answers)]
        return results

def minmax(arr):
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
