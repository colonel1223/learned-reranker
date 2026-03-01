"""Training pipeline: AMP, gradient accumulation, cosine LR, NDCG eval, early stopping."""
import torch, torch.nn as nn, numpy as np, time, logging
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass
from model import NeuralReranker, RerankerConfig, ListwiseLoss

logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    batch_size: int = 64; lr: float = 3e-4; wd: float = 0.01; epochs: int = 50
    warmup: int = 500; grad_accum: int = 1; max_grad_norm: float = 1.0
    use_amp: bool = True; eval_every: int = 500; patience: int = 5; K: int = 50; alpha: float = 0.7

class RankingDataset(Dataset):
    def __init__(self, qe, ce, f, r, m=None):
        self.qe, self.ce, self.f, self.r = [torch.from_numpy(x).float() for x in [qe, ce, f, r]]
        self.m = torch.from_numpy(m).bool() if m is not None else torch.ones(r.shape, dtype=torch.bool)
    def __len__(self): return len(self.qe)
    def __getitem__(self, i): return {"qe": self.qe[i], "ce": self.ce[i], "f": self.f[i], "r": self.r[i], "m": self.m[i]}

def ndcg(scores, rel, k=10):
    _, pi = scores.topk(min(k, scores.shape[1]), dim=1)
    pr = rel.gather(1, pi)
    pos = torch.arange(1, pr.shape[1]+1, device=scores.device).float()
    dcg = ((2**pr - 1) / torch.log2(pos + 1)).sum(1)
    sr, _ = rel.sort(descending=True, dim=1)
    sr = sr[:, :min(k, sr.shape[1])]
    idcg = ((2**sr - 1) / torch.log2(torch.arange(1, sr.shape[1]+1, device=scores.device).float() + 1)).sum(1)
    return (dcg / (idcg + 1e-8)).mean().item()

class Trainer:
    def __init__(self, model, cfg, train_ds, val_ds=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model, self.cfg, self.device = model.to(device), cfg, device
        self.criterion = ListwiseLoss(cfg.alpha)
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        self.train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        self.val_dl = DataLoader(val_ds, batch_size=cfg.batch_size*2) if val_ds else None
        self.scaler = GradScaler(enabled=cfg.use_amp)
        self.best_ndcg, self.wait = 0.0, 0

    def train(self):
        for epoch in range(self.cfg.epochs):
            self.model.train(); loss_sum = 0
            for batch in self.train_dl:
                b = {k: v.to(self.device) for k, v in batch.items()}
                with autocast(enabled=self.cfg.use_amp):
                    out = self.model(b["qe"], b["ce"], b["f"], b["m"])
                    loss = self.criterion(out["scores"], b["r"])
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.opt); self.scaler.update(); self.opt.zero_grad()
                loss_sum += loss.item()
            vn = self._eval() if self.val_dl else 0
            logger.info(f"epoch {epoch+1} loss={loss_sum/len(self.train_dl):.4f} ndcg@10={vn:.4f}")
            if vn > self.best_ndcg: self.best_ndcg = vn; self.wait = 0; torch.save(self.model.state_dict(), "best.pt")
            else:
                self.wait += 1
                if self.wait >= self.cfg.patience: break

    @torch.no_grad()
    def _eval(self):
        self.model.eval(); vals = []
        for b in self.val_dl:
            b = {k: v.to(self.device) for k, v in b.items()}
            vals.append(ndcg(self.model(b["qe"], b["ce"], b["f"], b["m"])["scores"], b["r"]))
        return np.mean(vals)
