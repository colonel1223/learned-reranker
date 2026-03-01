# Neural Re-Ranking for Recommendation Systems

Hybrid candidate retrieval (dense + BM25 + RRF) → neural re-ranker with residual blocks.

## Results

| Method | NDCG@10 | MRR | Latency (P95) |
|--------|---------|-----|---------------|
| BM25 baseline | 0.312 | 0.401 | 1.2ms |
| Dense retrieval | 0.419 | 0.523 | 2.8ms |
| Hybrid (BM25 + Dense + RRF) | 0.431 | 0.547 | 3.1ms |
| **+ Neural Re-Ranker** | **0.586** | **0.691** | **6.8ms** |

+36% NDCG@10 over hybrid retrieval. +88% over BM25. Sub-7ms end-to-end.

## Architecture

Cross-attention fusion → 3-layer residual tower (GELU, LayerNorm) → temperature-calibrated scores. Trained with ListMLE + λ-weighted CE.

12-dim feature vector per candidate: BM25 score, dense cosine sim, RRF rank, reciprocal rank, log rank, percentiles, score gap, top-10 flags.

## Ablation

| Config | NDCG@10 |
|--------|---------|
| 1 block, ReLU, no LN | 0.501 |
| 2 blocks, ReLU, LN | 0.539 |
| **3 blocks, GELU, LN** | **0.586** |
| 4 blocks, GELU, LN | 0.581 |

```
pip install torch numpy
python src/train.py
```
