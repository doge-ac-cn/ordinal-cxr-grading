"""P4.9c  +

1. 14(12+RAD-DINO u/w, seed0, cumulative): ->valQWK()->test(npz)
2. () vs top-3(val QWKtop-3, argmax) + bootstrap CI
 results/p49c_ensemble.json + results/test_predictions_release.npz
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score

PROJ = Path(__file__).resolve().parents[1]
LABEL_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right",
              "PleuralEffusion_Left", "PulmonaryOpacities_Right", "PulmonaryOpacities_Left",
              "Atelectasis_Right", "Atelectasis_Left"]
N_LEVELS = {"HeartSize": 4, **{c: 5 for c in LABEL_COLS if c != "HeartSize"}}
CFGS = [
    ("224bc_u", "features", "cls", False), ("224bc_w", "features", "cls", True),
    ("224bm_u", "features", "mean", False), ("224bm_w", "features", "mean", True),
    ("224lc_u", "features-large", "cls", False), ("224lc_w", "features-large", "cls", True),
    ("224lm_u", "features-large", "mean", False), ("224lm_w", "features-large", "mean", True),
    ("518bc_u", "features-518", "cls", False), ("518bc_w", "features-518", "cls", True),
    ("518bm_u", "features-518", "mean", False), ("518bm_w", "features-518", "mean", True),
    ("raddino_u", "features-raddino", "cls", False), ("raddino_w", "features-raddino", "cls", True),
]


def load_split(fdir, pool, split):
    key = "feats_mean" if pool == "mean" else "feats"
    fs, metas = [], []
    for f in sorted((PROJ / fdir).glob(f"{split}-*.npz")):
        fs.append(np.load(f)[key])
        metas.append(pd.read_parquet((PROJ / fdir) / f"{f.stem}.meta.parquet"))
    X = torch.from_numpy(np.concatenate(fs).astype(np.float32))
    m = pd.concat(metas, ignore_index=True)
    Y = torch.from_numpy(m[LABEL_COLS].to_numpy(np.int64))
    return X, Y, m


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


def train_cumul(Xtr, Ytr, Xva, Yva, seed=0, epochs=100, patience=10, weighted=False):
    torch.manual_seed(seed)
    model = CumHead(Xtr.shape[1], LABEL_COLS).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if weighted:
        pos_w = {}
        for ci, c in enumerate(LABEL_COLS):
            y = Ytr[:, ci].cuda()
            pw = []
            for t in range(N_LEVELS[c] - 1):
                pos = (y > t).sum().float().clamp(min=1)
                pw.append((len(Ytr) - pos) / pos)
            pos_w[c] = torch.stack(pw)
    bce = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    best, bad, best_state = -1, 0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            x = Xtr[idx].cuda()
            loss = 0.0
            for ci, c in enumerate(LABEL_COLS):
                y = Ytr[idx, ci].cuda()
                crit = nn.BCEWithLogitsLoss(pos_weight=pos_w[c]) if weighted else bce
                loss = loss + crit(model(x, c), cumulative_targets(y, N_LEVELS[c]))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = {c: torch.zeros(len(Xva), N_LEVELS[c] - 1) for c in LABEL_COLS}
            for jj in range(0, len(Xva), 16384):
                x = Xva[jj:jj + 16384].cuda()
                for ci, c in enumerate(LABEL_COLS):
                    vp[c][jj:jj + len(x)] = model(x, c).cpu()
            mq = np.mean([cohen_kappa_score(Yva[:, ci].numpy(),
                          (torch.sigmoid(vp[c]) > 0.5).sum(1).numpy(),
                          weights="quadratic", labels=list(range(N_LEVELS[c])))
                          for ci, c in enumerate(LABEL_COLS)])
        if mq > best:
            best, bad = float(mq), 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def probs_and_preds(model, X, bs=16384):
    """(,)(,).
    : unconstrained cumulativesigmoid, argmax()
    (-0.07 QWK), """
    probs = {c: torch.zeros(len(X), N_LEVELS[c]) for c in LABEL_COLS}
    hard = {c: torch.zeros(len(X), dtype=torch.int64) for c in LABEL_COLS}
    for i in range(0, len(X), bs):
        x = X[i:i + bs].cuda()
        for c in LABEL_COLS:
            out = model(x, c)
            cum = torch.sigmoid(out)
            hard[c][i:i + len(x)] = (cum > 0.5).sum(1).cpu()
            ext = torch.cat([torch.ones(len(x), 1, device=cum.device), cum,
                             torch.zeros(len(x), 1, device=cum.device)], dim=1)
            p = (ext[:, :-1] - ext[:, 1:]).clamp(min=0)
            probs[c][i:i + len(x)] = (p / p.sum(1, keepdim=True).clamp(min=1e-8)).cpu()
    return probs, {c: hard[c].numpy() for c in LABEL_COLS}


def main():
    t0 = time.time()
    cache = {}
    def get(fdir, pool):
        key = (fdir, pool)
        if key not in cache:
            cache[key] = (load_split(fdir, pool, "train"), load_split(fdir, pool, "val"),
                          load_split(fdir, pool, "test"))
        return cache[key]

    R, all_probs, all_counts = {}, {}, {}
    for tag, fdir, pool, w in CFGS:
        (Xtr, Ytr, _), (Xva, Yva, _), (Xte, Yte, _) = get(fdir, pool)
        model, vq = train_cumul(Xtr, Ytr, Xva, Yva, weighted=w)
        pv, pv_hard = probs_and_preds(model, Xva)
        pt, pt_hard = probs_and_preds(model, Xte)
        vqt = {c: float(cohen_kappa_score(Yva[:, ci].numpy(), pv_hard[c],
                    weights="quadratic", labels=list(range(N_LEVELS[c])))) for ci, c in enumerate(LABEL_COLS)}
        q = {c: float(cohen_kappa_score(Yte[:, ci].numpy(), pt_hard[c], weights="quadratic",
                    labels=list(range(N_LEVELS[c])))) for ci, c in enumerate(LABEL_COLS)}
        R[tag] = {"val_per_task": vqt, "test_mean": float(np.mean(list(q.values())))}
        all_probs[tag] = {"val": pv, "test": pt}
        all_counts[tag] = pt_hard
        print(f"[{tag}] val {vq:.4f} test {R[tag]['test_mean']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    sel_test, ens_test = {}, {}
    n_te = len(all_counts[CFGS[0][0]][LABEL_COLS[0]])
    for ci, c in enumerate(LABEL_COLS):
        ranked = sorted(R, key=lambda k: -R[k]["val_per_task"][c])
        best = ranked[0]
        sel_test[c] = all_counts[best][c]
        top3 = ranked[:3]
        ens_test[c] = np.rint(np.mean([all_counts[k][c].astype(np.float64) for k in top3], axis=0)).astype(np.int64)
    single_q = [cohen_kappa_score(Yte[:, ci].numpy(), sel_test[c], weights="quadratic",
                                  labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]
    ens_q = [cohen_kappa_score(Yte[:, ci].numpy(), ens_test[c], weights="quadratic",
                               labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]
    Yte_last = get(CFGS[-1][1], CFGS[-1][2])[2][1].numpy()
    single_mean, ens_mean = float(np.mean(single_q)), float(np.mean(ens_q))

    rng = np.random.default_rng(0)
    boots = np.empty(1000)
    for b in range(1000):
        idx = rng.integers(0, n_te, n_te)
        qw = [cohen_kappa_score(Yte_last[idx, ci], ens_test[c][idx], weights="quadratic",
                                labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]
        boots[b] = float(np.mean(qw))
    ci = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]

    np.savez_compressed(PROJ / "results" / "test_predictions_release.npz",
                        **{f"{c}/{tag}": all_counts[tag][c]
                           for tag in all_counts for c in LABEL_COLS},
                        **{f"y_true/{c}": Yte_last[:, ci] for ci, c in enumerate(LABEL_COLS)})
    ens_lk = [cohen_kappa_score(Yte[:, ci].numpy(), ens_test[c], weights="linear",
                                labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]
    ens_ea = [float((Yte[:, ci].numpy() == ens_test[c]).mean()) for ci, c in enumerate(LABEL_COLS)]
    out = {"headline_single_best": {"mean_qwk": single_mean},
           "headline_top3_ensemble": {"mean_qwk": ens_mean, "bootstrap95ci": ci,
                                      "mean_linear_kappa": round(float(np.mean(ens_lk)), 4),
                                      "mean_exact_acc": round(float(np.mean(ens_ea)), 4)},
           "val_per_task_matrix": {k: R[k]["val_per_task"] for k in R},
           "per_task_single": {c: round(float(single_q[ci]), 4) for ci, c in enumerate(LABEL_COLS)},
           "per_task_ensemble": {c: round(float(ens_q[ci]), 4) for ci, c in enumerate(LABEL_COLS)},
           "configs_test_mean": {k: round(v["test_mean"], 4) for k, v in R.items()}}
    (PROJ / "results" / "p49c_ensemble.json").write_text(json.dumps(out, indent=1))
    print(f" {single_mean:.4f} | top-3 {ens_mean:.4f} CI{ci}", flush=True)
    print(" results/p49c_ensemble.json + test_predictions_release.npz", flush=True)


if __name__ == "__main__":
    main()
