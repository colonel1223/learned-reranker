"""Eval suite: NDCG, MAP, MRR, P@K, R@K, latency profiling."""
import numpy as np, time, torch
from dataclasses import dataclass

@dataclass
class EvalResult:
    ndcg5: float; ndcg10: float; map_: float; mrr: float; p5: float; p10: float
    lat_p50: float; lat_p95: float; lat_p99: float
    def __repr__(self):
        return (f"NDCG@5={self.ndcg5:.4f} NDCG@10={self.ndcg10:.4f} MAP={self.map_:.4f} "
                f"MRR={self.mrr:.4f} P@5={self.p5:.4f} P@10={self.p10:.4f} "
                f"lat_p50={self.lat_p50:.2f}ms lat_p95={self.lat_p95:.2f}ms lat_p99={self.lat_p99:.2f}ms")

def ndcg_at_k(s, r, k):
    idx = np.argsort(-s)[:k]; pr = r[idx]
    pos = np.arange(1, len(pr)+1); dcg = np.sum((2**pr - 1) / np.log2(pos + 1))
    ir = np.sort(r)[::-1][:k]; idcg = np.sum((2**ir - 1) / np.log2(np.arange(1, len(ir)+1) + 1))
    return dcg / max(idcg, 1e-8)

def ap(s, r, th=1.0):
    order = np.argsort(-s); sr = (r[order] >= th).astype(float)
    if sr.sum() == 0: return 0.0
    cs = np.cumsum(sr); pai = cs / np.arange(1, len(sr)+1)
    return np.sum(pai * sr) / sr.sum()

def rr(s, r, th=1.0):
    for i, idx in enumerate(np.argsort(-s), 1):
        if r[idx] >= th: return 1.0/i
    return 0.0

@torch.no_grad()
def evaluate(model, dl, device="cuda", warmup=10):
    model.eval()
    n5, n10, maps, mrrs, p5s, p10s, lats = [], [], [], [], [], [], []
    for i, b in enumerate(dl):
        b = {k: v.to(device) for k, v in b.items()}
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(b["qe"], b["ce"], b["f"], b.get("m"))
        if device == "cuda": torch.cuda.synchronize()
        if i >= warmup: lats.append((time.perf_counter() - t0) * 1000)
        sc, rl = out["scores"].cpu().numpy(), b["r"].cpu().numpy()
        for j in range(sc.shape[0]):
            s, r = sc[j], rl[j]
            n5.append(ndcg_at_k(s, r, 5)); n10.append(ndcg_at_k(s, r, 10))
            maps.append(ap(s, r)); mrrs.append(rr(s, r))
            top5 = np.argsort(-s)[:5]; top10 = np.argsort(-s)[:10]
            p5s.append(sum(r[top5] >= 1) / 5); p10s.append(sum(r[top10] >= 1) / 10)
    la = np.array(lats) if lats else np.array([0.0])
    return EvalResult(np.mean(n5), np.mean(n10), np.mean(maps), np.mean(mrrs),
                      np.mean(p5s), np.mean(p10s), np.percentile(la,50), np.percentile(la,95), np.percentile(la,99))
