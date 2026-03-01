"""
Hybrid Candidate Retrieval: Dense + BM25 + Reciprocal Rank Fusion.
Author: Spencer Cottrell
"""
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class Document:
    doc_id: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class RetrievalResult:
    doc_id: str
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    combined_score: float = 0.0
    rank: int = 0

class BM25:
    """Okapi BM25 (Robertson & Zaragoza 2009). k1=1.2, b=0.75."""
    def __init__(self, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_lens, self.tf, self.df = {}, {}, defaultdict(int)

    def index(self, docs):
        self.doc_count = len(docs)
        total = 0
        for d in docs:
            toks = d.text.lower().split()
            self.doc_lens[d.doc_id] = len(toks)
            total += len(toks)
            freq = defaultdict(int)
            for t in toks: freq[t] += 1
            self.tf[d.doc_id] = dict(freq)
            for t in set(toks): self.df[t] += 1
        self.avg_dl = total / max(self.doc_count, 1)

    def score(self, query, doc_id):
        s = 0.0
        dtf = self.tf.get(doc_id, {})
        dl = self.doc_lens.get(doc_id, 0)
        for t in query.lower().split():
            if t not in dtf: continue
            tf = dtf[t]
            idf = max(math.log((self.doc_count - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1), 0)
            s += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
        return s

    def retrieve(self, query, top_k=100):
        scores = [(did, self.score(query, did)) for did in self.tf]
        return sorted([s for s in scores if s[1] > 0], key=lambda x: -x[1])[:top_k]

class DenseRetriever:
    def __init__(self): self.embs = {}
    def index(self, docs):
        for d in docs:
            if d.embedding is not None:
                n = np.linalg.norm(d.embedding)
                if n > 0: self.embs[d.doc_id] = d.embedding / n
    def retrieve(self, qe, top_k=100):
        n = np.linalg.norm(qe)
        if n == 0: return []
        q = qe / n
        return sorted([(did, float(np.dot(q, e))) for did, e in self.embs.items()], key=lambda x: -x[1])[:top_k]

def rrf(*lists, k=60):
    scores = defaultdict(float)
    for rl in lists:
        for rank, (did, _) in enumerate(rl):
            scores[did] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])

class HybridRetriever:
    def __init__(self, dw=0.6, sw=0.4):
        self.bm25, self.dense = BM25(), DenseRetriever()
        self.dw, self.sw = dw, sw
        self.documents = {}
    def index(self, docs):
        self.bm25.index(docs); self.dense.index(docs)
        for d in docs: self.documents[d.doc_id] = d
    def retrieve(self, query, query_emb, top_k=50):
        fk = top_k * 3
        sr = self.bm25.retrieve(query, fk)
        dr = self.dense.retrieve(query_emb, fk)
        fused = rrf(sr, dr)
        ss = {d: s for d, s in sr}
        ds = {d: s for d, s in dr}
        return [RetrievalResult(doc_id=did, dense_score=ds.get(did, 0), bm25_score=ss.get(did, 0),
                rrf_score=rs, combined_score=self.dw * ds.get(did, 0) + self.sw * ss.get(did, 0), rank=i+1)
                for i, (did, rs) in enumerate(fused[:top_k])]
