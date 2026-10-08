"""TAIX-Ray  v0.1 —— DINOv2


  1) CORAL  (Cao et al. 2020): logits + BCE,
  2) Plain-CE :  softmax

: QWK (quadratic weighted kappa) / MAE / ,  val + test
: python train_ordinal.py [--epochs 100] [--seed 0]
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
LABEL_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right",
              "PleuralEffusion_Left", "PulmonaryOpacities_Right", "PulmonaryOpacities_Left",
              "Atelectasis_Right", "Atelectasis_Left"]
N_LEVELS = {"HeartSize": 4, **{c: 5 for c in LABEL_COLS if c != "HeartSize"}}


def load_split(split, fdir="features", pool="cls"):
    key = "feats_mean" if pool == "mean" else "feats"
    fs, metas = [], []
    for f in sorted((ROOT / fdir).glob(f"{split}-*.npz")):
        z = np.load(f)
        if key not in z:
            raise KeyError(f"{f.name}  {key}")
        fs.append(z[key])
        metas.append(pd.read_parquet((ROOT / fdir) / f"{f.stem}.meta.parquet"))
    X = torch.from_numpy(np.concatenate(fs).astype(np.float32))
    m = pd.concat(metas, ignore_index=True)
    Y = torch.from_numpy(m[LABEL_COLS].to_numpy(np.int64))
    return X, Y, m


def cumulative_targets(y, k):
    """y:(N,) 0..k-1 -> (N,k-1) """
    t = torch.arange(k - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > t).float()


class CoralHead(nn.Module):
    """(768->k-1), CORAL(+)"""
    def __init__(self, dim, cols):
        super().__init__()
        self.cols = cols
        self.heads = nn.ModuleDict({c: nn.Linear(dim, N_LEVELS[c] - 1) for c in cols})

    def forward(self, x, col):
        return self.heads[col](x)


class CoralSharedHead(nn.Module):
    """CORAL (Cao et al. 2020): w, (k-1) -> logits = w·x + b_k"""
    def __init__(self, dim, cols):
        super().__init__()
        self.cols = cols
        self.w = nn.ModuleDict({c: nn.Linear(dim, 1) for c in cols})
        self.b = nn.ParameterDict({c: nn.Parameter(torch.zeros(N_LEVELS[c] - 1)) for c in cols})

    def forward(self, x, col):
        return self.w[col](x) + self.b[col]


class CeHead(nn.Module):
    def __init__(self, dim, cols):
        super().__init__()
        self.cols = cols
        self.heads = nn.ModuleDict({c: nn.Linear(dim, N_LEVELS[c]) for c in cols})

    def forward(self, x, col):
        return self.heads[col](x)


def corn_task_loss(logits, y, k_levels):
    """CORN: k y>k-1  P(y>k|y>k-1)"""
    losses = []
    for k in range(k_levels - 1):
        if k == 0:
            mask = torch.ones(len(y), dtype=torch.bool, device=y.device)
            targets = (y > 0).float()
        else:
            mask = y > (k - 1)
            targets = (y[mask] > k).float()
        if mask.sum() == 0:
            continue
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits[mask, k], targets))
    return sum(losses) / max(len(losses), 1)


def predict_corn(logits):
    """ -> P(y>k), """
    cum = torch.cumprod(torch.sigmoid(logits), dim=1)
    return (cum > 0.5).sum(1)


def soft_ordinal_targets(y, k, lam=0.3, tau=1.0):
    """CE: (1-λ)·onehot + λ·(), """
    d = torch.arange(k, device=y.device).unsqueeze(0) - y.unsqueeze(1)
    g = torch.softmax(-d.float() ** 2 / tau, dim=1)
    onehot = torch.zeros_like(g).scatter_(1, y.unsqueeze(1), 1.0)
    t = (1 - lam) * onehot + lam * g
    return t / t.sum(1, keepdim=True)


def predict_coral(logits):
    """>0.5 = """
    return (torch.sigmoid(logits) > 0.5).sum(1)


def evaluate(model, kind, X, Y, bs=8192):
    model.eval()
    preds = {c: [] for c in LABEL_COLS}
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = X[i:i + bs].to("cuda")
            for c in LABEL_COLS:
                out = model(x, c)
                preds[c].append((predict_corn(out) if kind == "corn"
                                 else predict_coral(out) if kind in ("coral", "cumul", "coral_shared")
                                 else out.argmax(1)).cpu())
    metrics = {}
    for ci, c in enumerate(LABEL_COLS):
        p, y = torch.cat(preds[c]).numpy(), Y[:, ci].numpy()
        metrics[c] = {
            "qwk": cohen_kappa_score(y, p, weights="quadratic", labels=list(range(N_LEVELS[c]))),
            "mae": float(np.mean(np.abs(y - p))),
            "exact": float((y == p).mean()),
        }
    return metrics


def train_one(kind, Xtr, Ytr, Xva, Yva, epochs, seed, lr=1e-3, wd=1e-4, weighted=False):
    torch.manual_seed(seed)
    arch = {"coral": CoralHead, "coral_shared": CoralSharedHead, "corn": CoralHead,
            "ce": CeHead, "softord": CeHead, "cumul": CoralHead}[kind]
    model = arch(Xtr.shape[1], LABEL_COLS).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    if weighted:
        ce_w, pos_w = {}, {}
        for i, c in enumerate(LABEL_COLS):
            y = Ytr[:, i].to("cuda")
            cnt = torch.bincount(y, minlength=N_LEVELS[c]).float().clamp(min=1)
            w = cnt.sum() / (N_LEVELS[c] * cnt)
            ce_w[c] = w / w.mean()
            pw = []
            for t in range(N_LEVELS[c] - 1):
                pos = (y > t).sum().float().clamp(min=1)
                pw.append((len(Ytr) - pos) / pos)
            pos_w[c] = torch.stack(pw)

    n = len(Xtr)
    bs = 4096
    best_qwk, best_state, patience, bad = -1, None, 10, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x = Xtr[idx].to("cuda")
            loss = 0.0
            for ci, c in enumerate(LABEL_COLS):
                y = Ytr[idx, ci].to("cuda")
                out = model(x, c)
                if kind == "corn":
                    loss = loss + corn_task_loss(out, y, N_LEVELS[c])
                elif kind in ("coral", "cumul", "coral_shared"):
                    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w[c]) if weighted else bce
                    loss = loss + crit(out, cumulative_targets(y, N_LEVELS[c]))
                elif kind == "softord":
                    yt = y
                    if weighted:
                        yt = yt
                    logp = torch.log_softmax(out, dim=1)
                    loss = loss + -(soft_ordinal_targets(yt, N_LEVELS[c]) * logp).sum(1).mean()
                else:
                    crit = nn.CrossEntropyLoss(weight=ce_w[c]) if weighted else ce
                    loss = loss + crit(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        m = evaluate(model, kind, Xva, Yva)
        mq = float(np.mean([m[c]["qwk"] for c in LABEL_COLS]))
        if mq > best_qwk:
            best_qwk, bad = mq, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_qwk, time.time() - t0, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="")
    ap.add_argument("--cls-weight", action="store_true", help="P1.2 ")
    ap.add_argument("--loss-matrix", action="store_true", help="P2.2 4")
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--pool", default="cls", choices=["cls", "mean"])
    args = ap.parse_args()

    print("...", flush=True)
    Xtr, Ytr, _ = load_split("train", args.features_dir, args.pool)
    Xva, Yva, _ = load_split("val", args.features_dir, args.pool)
    Xte, Yte, _ = load_split("test", args.features_dir, args.pool)
    print(f"train {tuple(Xtr.shape)} | val {tuple(Xva.shape)} | test {tuple(Xte.shape)}", flush=True)

    results = {}
    kinds = ["ce", "coral"] if not args.loss_matrix else ["ce", "coral", "coral_shared", "softord", "corn"]
    for kind in kinds:
        model, vqwk, secs, eps = train_one(kind, Xtr, Ytr, Xva, Yva, args.epochs, args.seed,
                                           weighted=args.cls_weight)
        va = evaluate(model, kind, Xva, Yva)
        te = evaluate(model, kind, Xte, Yte)
        results[kind] = {
            "val": va, "test": te,
            "val_mean_qwk": float(np.mean([va[c]["qwk"] for c in LABEL_COLS])),
            "test_mean_qwk": float(np.mean([te[c]["qwk"] for c in LABEL_COLS])),
            "train_seconds": round(secs, 1), "epochs_run": eps,
        }
        print(f"\n=== {kind.upper()} (val mean QWK {results[kind]['val_mean_qwk']:.4f}, "
              f"test mean QWK {results[kind]['test_mean_qwk']:.4f}, {secs:.0f}s, {eps}ep) ===", flush=True)
        for c in LABEL_COLS:
            print(f"  {c:26s} val QWK {va[c]['qwk']:.3f} MAE {va[c]['mae']:.3f} | "
                  f"test QWK {te[c]['qwk']:.3f} MAE {te[c]['mae']:.3f}", flush=True)

    suffix = f"_{args.tag}" if args.tag else ""
    out = ROOT / "results" / f"linear_probe_seed{args.seed}{suffix}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\n {out}", flush=True)


if __name__ == "__main__":
    main()
