"""P2.4  + ,

 518-base-mean  CE  cumulative (seed0, ),  test :
  1) ECE(15bin) +
  2) /
  3) split-conformal: val(90%) -> test + ()
 results/p24_calibration.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
def class_probs(model, kind, X, Y, bs=16384):
    """(N,k)"""
    probs = {c: torch.zeros(len(X), N_LEVELS[c]) for c in LABEL_COLS}
    for i in range(0, len(X), bs):
        x = X[i:i + bs].cuda()
        for c in LABEL_COLS:
            out = model(x, c)
            if kind == "cumul":
                k = N_LEVELS[c]
                cum = torch.sigmoid(out)  # P(y>t), t=0..k-2
                ones = torch.ones(len(x), 1, device=cum.device)
                ext = torch.cat([ones, cum, torch.zeros(len(x), 1, device=cum.device)], dim=1)
                p = (ext[:, :-1] - ext[:, 1:]).clamp(min=0)
                probs[c][i:i + len(x)] = (p / p.sum(1, keepdim=True).clamp(min=1e-8)).cpu()
            else:
                probs[c][i:i + len(x)] = torch.softmax(out, dim=1).cpu()
    return probs


def ece(p_true, p_pred, n_bins=15):
    conf = p_pred.max(1)
    pred = conf.indices
    correct = (pred == p_true).float()
    bins = torch.clamp((conf.values * n_bins).long(), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.sum() > 0:
            e += (correct[m].mean() - conf.values[m].mean()).abs() * m.float().mean()
    return float(e)


def ordinal_interval_set(p, mass=1 - ALPHA):
    """>=mass, p:(N,k)"""
    N, k = p.shape
    l = torch.zeros(N, dtype=torch.long)
    r = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        best = (k, 0)
        for w in range(1, k + 1):
            csum = torch.tensor([p[i, s:s + w].sum() for s in range(k - w + 1)])
            s = int(csum.argmax())
            if csum[s] >= mass:
                best = (s, s + w - 1)
                break
        l[i], r[i] = best
    return l, r


def main():
    print("518...", flush=True)
    Xtr, Ytr = load_split("train")
    Xva, Yva = load_split("val")
    Xte, Yte = load_split("test")
    print(f"train {tuple(Xtr.shape)} test {tuple(Xte.shape)}", flush=True)

    out = {"config": "518-base-mean, seed0, ; conformal: split, val, 90%, "}
    for kind, arch in [("cumul", CumHead), ("ce", CeHead)]:
        model = train_head(arch, Xtr, Ytr)
        pv = class_probs(model, kind, Xva, Yva)
        pt = class_probs(model, kind, Xte, Yte)
        res = {"ece": {}, "per_grade_recall": {}, "conformal": {}}
        for ci, c in enumerate(LABEL_COLS):
            k = N_LEVELS[c]
            yt = Yte[:, ci]
            res["ece"][c] = round(ece(yt, pt[c]), 4)
            pred = pt[c].argmax(1)
            rec = {}
            for g in range(k):
                m = yt == g
                rec[str(g)] = round(float((pred[m] == g).float().mean()), 3) if m.sum() else None
            res["per_grade_recall"][c] = rec
            lv, rv = ordinal_interval_set(pv[c].numpy(), 1 - ALPHA)
            errs = (Yva[:, ci] < lv.float() - 0.5) | (Yva[:, ci] > rv.float() + 0.5)
            mass_q = torch.quantile(errs.float(), min(1 - ALPHA + 0.02, 1.0))
            lt, rt = ordinal_interval_set(pt[c].numpy(), 1 - float(mass_q))
            cover = float(((Yte[:, ci] >= lt - 0.5) & (Yte[:, ci] <= rt + 0.5)).float().mean())
            res["conformal"][c] = {"coverage": round(cover, 4),
                                   "mean_set_size": round(float((rt - lt + 1).float().mean()), 3)}
        out[kind] = res
        print(f"{kind}: ECE {np.mean(list(res['ece'].values())):.4f} | "
              f"90%:  {np.mean([v['coverage'] for v in res['conformal'].values()]):.4f}, "
              f" {np.mean([v['mean_set_size'] for v in res['conformal'].values()]):.2f}", flush=True)

    (PROJ / "results" / "p24_calibration.json").write_text(json.dumps(out, indent=1))
    print(" results/p24_calibration.json", flush=True)


if __name__ == "__main__":
    main()
