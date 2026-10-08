"""P4.1 split conformal (CIF, Lu et al.-style ordinal interval fusion)

(P2.4):
  1. conformity score: argmax,
  2. : Q = sorted_scores[ceil((n+1)*(1-alpha))-1] (fudge)
  3. =argmax>=Q -> >=1-alphasplit conformal,
     (softmax); ECE
518-base-mean cumulative  CE ,  results/p41_conformal_proper.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score

PROJ = Path(__file__).resolve().parents[1]
FEAT = PROJ / "features-518"
LABEL_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right",
              "PleuralEffusion_Left", "PulmonaryOpacities_Right", "PulmonaryOpacities_Left",
              "Atelectasis_Right", "Atelectasis_Left"]
N_LEVELS = {"HeartSize": 4, **{c: 5 for c in LABEL_COLS if c != "HeartSize"}}
ALPHA = 0.10


def load_split(split):
    fs, metas = [], []
    for f in sorted(FEAT.glob(f"{split}-*.npz")):
        fs.append(np.load(f)["feats_mean"])
        metas.append(pd.read_parquet(FEAT / f"{f.stem}.meta.parquet"))
    X = torch.from_numpy(np.concatenate(fs).astype(np.float32))
    Y = torch.from_numpy(pd.concat(metas, ignore_index=True)[LABEL_COLS].to_numpy(np.int64))
    return X, Y


def cumulative_targets(y, k):
    t = torch.arange(k - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > t).float()


class CumHead(nn.Module):
    def __init__(self, dim, cols):
        super().__init__()
        self.cols = cols
        self.heads = nn.ModuleDict({c: nn.Linear(dim, N_LEVELS[c] - 1) for c in cols})

    def forward(self, x, col):
        return self.heads[col](x)


class CeHead(nn.Module):
    def __init__(self, dim, cols):
        super().__init__()
        self.cols = cols
        self.heads = nn.ModuleDict({c: nn.Linear(dim, N_LEVELS[c]) for c in cols})

    def forward(self, x, col):
        return self.heads[col](x)


def train_head(arch, Xtr, Ytr, seed=0, epochs=100):
    torch.manual_seed(seed)
    model = arch(Xtr.shape[1], LABEL_COLS).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            x = Xtr[idx].cuda()
            loss = 0.0
            for ci, c in enumerate(LABEL_COLS):
                y = Ytr[idx, ci].cuda()
                out = model(x, c)
                loss = loss + (bce(out, cumulative_targets(y, N_LEVELS[c])) if arch is CumHead
                               else ce(out, y))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def class_probs(model, kind, X, bs=16384):
    probs = {c: torch.zeros(len(X), N_LEVELS[c]) for c in LABEL_COLS}
    for i in range(0, len(X), bs):
        x = X[i:i + bs].cuda()
        for c in LABEL_COLS:
            out = model(x, c)
            if kind == "cumul":
                k = N_LEVELS[c]
                cum = torch.sigmoid(out)
                ext = torch.cat([torch.ones(len(x), 1, device=cum.device), cum,
                                 torch.zeros(len(x), 1, device=cum.device)], dim=1)
                p = (ext[:, :-1] - ext[:, 1:]).clamp(min=0)
                probs[c][i:i + len(x)] = (p / p.sum(1, keepdim=True).clamp(min=1e-8)).cpu()
            else:
                probs[c][i:i + len(x)] = torch.softmax(out, dim=1).cpu()
    return probs


def _grow(p_row, am, target=None, q=None):
    """argmax(). target: ;
    q: >=q. ."""
    k = len(p_row)
    lo = hi = int(am)
    m = float(p_row[lo])
    if target is not None and lo <= target <= hi:
        return m
    while True:
        left_ok = lo - 1 >= 0
        right_ok = hi + 1 < k
        if not left_ok and not right_ok:
            m = 1.0
            break
        if left_ok and right_ok and p_row[lo - 1] >= p_row[hi + 1]:
            lo -= 1; m += float(p_row[lo])
        elif left_ok:
            lo -= 1; m += float(p_row[lo])
        else:
            hi += 1; m += float(p_row[hi])
        if target is not None and lo <= target <= hi:
            return m
        if q is not None and m >= q:
            return lo, hi, hi - lo + 1
    return m


def cif_mass(p, y):
    """conformity score: """
    N = p.shape[0]
    return np.array([_grow(p[i], int(p[i].argmax()), target=int(y[i])) for i in range(N)])


def cif_sets(p, q):
    """>=q(score)"""
    N = p.shape[0]
    sizes = np.zeros(N, dtype=int)
    lo_all = np.zeros(N, dtype=int)
    hi_all = np.zeros(N, dtype=int)
    for i in range(N):
        lo_all[i], hi_all[i], sizes[i] = _grow(p[i], int(p[i].argmax()), q=q)
    return lo_all, hi_all, sizes


def ece(p_true_counts, p_pred, n_bins=15):
    conf = p_pred.max(1)
    pred = conf.indices
    correct = (pred == p_true_counts).float()
    bins = torch.clamp((conf.values * n_bins).long(), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.sum() > 0:
            e += (correct[m].mean() - conf.values[m].mean()).abs() * m.float().mean()
    return float(e)


def main():
    print("518...", flush=True)
    Xtr, Ytr = load_split("train")
    Xva, Yva = load_split("val")
    Xte, Yte = load_split("test")

    out = {"protocol": ("proper split conformal, CIF-style ordinal conformity score "
                        "(alternating growth from argmax, score=mass at true-label inclusion), "
                        f"finite-sample quantile ceil((n+1)(1-alpha))/n, alpha={ALPHA}, "
                        "calibration=val, evaluation=test, config=518-base-mean seed0"),
           "heads": {}}
    for kind, arch in [("cumul", CumHead), ("ce", CeHead)]:
        model = train_head(arch, Xtr, Ytr)
        pv = class_probs(model, kind, Xva)
        pt = class_probs(model, kind, Xte)
        head_out = {"ece": {}, "conformal": {}}
        for ci, c in enumerate(LABEL_COLS):
            k = N_LEVELS[c]
            yv, yt = Yva[:, ci].numpy(), Yte[:, ci].numpy()
            head_out["ece"][c] = round(ece(Yte[:, ci], pt[c]), 4)
            sv = cif_mass(pv[c].numpy(), yv)
            n = len(sv)
            q_idx = min(int(np.ceil((n + 1) * (1 - ALPHA))) - 1, n - 1)
            q = float(np.sort(sv)[q_idx])
            lt, rt, sizes = cif_sets(pt[c].numpy(), min(q, 1.0))
            cover = float(np.mean((yt >= lt) & (yt <= rt)))
            head_out["conformal"][c] = {
                "q": round(min(q, 1.0), 4),
                "coverage": round(cover, 4),
                "mean_set_size": round(float(sizes.mean()), 3),
                "median_set_size": int(np.median(sizes)),
            }
            print(f"  {kind}/{c}: ECE {head_out['ece'][c]:.4f} | cov {cover:.3f} "
                  f"| size {sizes.mean():.2f}", flush=True)
        head_out["mean_ece"] = round(float(np.mean(list(head_out["ece"].values()))), 4)
        head_out["mean_coverage"] = round(float(np.mean([v["coverage"] for v in head_out["conformal"].values()])), 4)
        head_out["mean_size"] = round(float(np.mean([v["mean_set_size"] for v in head_out["conformal"].values()])), 3)
        out["heads"][kind] = head_out
        print(f"{kind}: meanECE {head_out['mean_ece']} | meanCov {head_out['mean_coverage']} "
              f"| meanSize {head_out['mean_size']}", flush=True)

    (PROJ / "results" / "p41_conformal_proper.json").write_text(json.dumps(out, indent=1))
    print(" results/p41_conformal_proper.json", flush=True)


if __name__ == "__main__":
    main()
