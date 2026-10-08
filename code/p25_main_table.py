"""P2.5 : Fold0() -> GroupKFold(5) by PatientID + bootstrap 95% CI

1) 12(seed0), val ->  test mean QWK + bootstrap 95% CI (1000)
2) (518bm cumul) GroupKFold(5): 4/1 -> test mean±std
 results/p25_main_table.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold

PROJ = Path(__file__).resolve().parents[1]
LABEL_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right",
              "PleuralEffusion_Left", "PulmonaryOpacities_Right", "PulmonaryOpacities_Left",
              "Atelectasis_Right", "Atelectasis_Left"]
N_LEVELS = {"HeartSize": 4, **{c: 5 for c in LABEL_COLS if c != "HeartSize"}}
CFGS = [  # (tag, fdir, pool, weighted)
    ("224bc_u", "features", "cls", False), ("224bc_w", "features", "cls", True),
    ("224bm_u", "features", "mean", False), ("224bm_w", "features", "mean", True),
    ("224lc_u", "features-large", "cls", False), ("224lc_w", "features-large", "cls", True),
    ("224lm_u", "features-large", "mean", False), ("224lm_w", "features-large", "mean", True),
    ("518bc_u", "features-518", "cls", False), ("518bc_w", "features-518", "cls", True),
    ("518bm_u", "features-518", "mean", False), ("518bm_w", "features-518", "mean", True),
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


def train_cumul(Xtr, Ytr, Xva, Yva, seed=0, epochs=100, patience=10, lr=1e-3, wd=1e-4, weighted=False):
    torch.manual_seed(seed)
    model = CumHead(Xtr.shape[1], LABEL_COLS).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
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
        mq = qwk_mean(model, Xva, Yva)
        if mq > best:
            best, bad = mq, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def predict(model, X, bs=16384):
    model.eval()
    out = {c: torch.zeros(len(X), N_LEVELS[c] - 1) for c in LABEL_COLS}
    for i in range(0, len(X), bs):
        x = X[i:i + bs].cuda()
        for c in LABEL_COLS:
            out[c][i:i + len(x)] = model(x, c).cpu()
    preds = {c: (torch.sigmoid(v) > 0.5).sum(1) for c, v in out.items()}
    return preds


def qwk_mean(model, X, Y):
    p = predict(model, X)
    return float(np.mean([cohen_kappa_score(Y[:, ci].numpy(), p[c].numpy(), weights="quadratic",
                                            labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]))


def main():
    t0 = time.time()
    cache = {}

    def get(fdir, pool):
        key = (fdir, pool)
        if key not in cache:
            cache[key] = (load_split(fdir, pool, "train"), load_split(fdir, pool, "val"),
                          load_split(fdir, pool, "test"))
        return cache[key]

    R, preds_sel = {}, {}
    for tag, fdir, pool, w in CFGS:
        (Xtr, Ytr, _), (Xva, Yva, _), (Xte, Yte, _) = get(fdir, pool)
        model, vq = train_cumul(Xtr, Ytr, Xva, Yva, weighted=w)
        pv = predict(model, Xva)
        vqt = {c: float(cohen_kappa_score(Yva[:, ci].numpy(), pv[c].numpy(), weights="quadratic",
                                          labels=list(range(N_LEVELS[c])))) for ci, c in enumerate(LABEL_COLS)}
        pt = predict(model, Xte)
        q = {c: cohen_kappa_score(Yte[:, ci].numpy(), pt[c].numpy(), weights="quadratic",
                                  labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)}
        lk = {c: cohen_kappa_score(Yte[:, ci].numpy(), pt[c].numpy(), weights="linear",
                                   labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)}
        ea = {c: float((Yte[:, ci].numpy() == pt[c].numpy()).mean()) for ci, c in enumerate(LABEL_COLS)}
        R[tag] = {"val": vq, "val_per_task": vqt,
                  "test_qwk": {c: float(q[c]) for c in LABEL_COLS},
                  "test_linear_kappa": {c: float(lk[c]) for c in LABEL_COLS},
                  "test_exact_acc": {c: float(ea[c]) for c in LABEL_COLS},
                  "test_mean": float(np.mean(list(q.values()))),
                  "test_mean_linear_kappa": float(np.mean(list(lk.values()))),
                  "test_mean_exact_acc": float(np.mean(list(ea.values())))}
        preds_sel[tag] = pt
        print(f"[{tag}] val {vq:.4f} test {R[tag]['test_mean']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    sel, sel_test = {}, {}
    for ci, c in enumerate(LABEL_COLS):
        best = max(R, key=lambda k: R[k]["val_per_task"][c])
        sel[c] = best
        sel_test[c] = preds_sel[best][c].numpy()
    head_qwk = np.array([cohen_kappa_score(Yte[:, ci].numpy(), sel_test[c], weights="quadratic",
                                           labels=list(range(N_LEVELS[c])))
                         for ci, c in enumerate(LABEL_COLS)])
    head_lk = np.array([cohen_kappa_score(Yte[:, ci].numpy(), sel_test[c], weights="linear",
                                          labels=list(range(N_LEVELS[c])))
                        for ci, c in enumerate(LABEL_COLS)])
    head_ea = np.array([float((Yte[:, ci].numpy() == sel_test[c]).mean()) for ci, c in enumerate(LABEL_COLS)])
    headline = float(head_qwk.mean())
    n_te = len(sel_test[LABEL_COLS[0]])
    rng = np.random.default_rng(0)
    boots = np.empty(1000)
    for b in range(1000):
        idx = rng.integers(0, n_te, n_te)
        qw = [cohen_kappa_score(Yte[idx, ci].numpy(), sel_test[c][idx], weights="quadratic",
                                labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]
        boots[b] = float(np.mean(qw))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    print(f": {headline:.4f} bootstrap95%CI {ci}", flush=True)

    # --- 3) GroupKFold(5) by PatientID (train+val, 518bm cumul) ---
    (Xtr, Ytr, mtr), (Xva, Yva, mva), (Xte, Yte, _) = get("features-518", "mean")
    Xall = torch.cat([Xtr, Xva]); Yall = torch.cat([Ytr, Yva])
    mall = pd.concat([mtr, mva], ignore_index=True)
    pid_map = {p: i for i, p in enumerate(mall["PatientID"].unique())}
    groups = mall["PatientID"].map(pid_map).to_numpy()
    gkf = GroupKFold(n_splits=5)
    cv = []
    for fi, (tri, vai) in enumerate(gkf.split(Xall, groups=groups)):
        model, vq = train_cumul(Xall[tri], Yall[tri], Xall[vai], Yall[vai], epochs=60, patience=8, weighted=False)
        mq = qwk_mean(model, Xte, Yte)
        cv.append({"fold": fi, "n_train": int(len(tri)), "val_qwk": round(vq, 4),
                   "test_qwk": round(mq, 4)})
        print(f"fold{fi}: val {vq:.4f} test {mq:.4f}", flush=True)
    cv_mean = float(np.mean([f["test_qwk"] for f in cv]))
    cv_std = float(np.std([f["test_qwk"] for f in cv], ddof=1))

    out = {"fold_semantics": "Fold0() -> GroupKFold(5) by PatientID on train+val, test",
           "headline": {"test_mean_qwk": headline, "bootstrap95ci": ci, "n_boot": 1000,
                        "test_mean_linear_kappa": float(head_lk.mean()),
                        "test_mean_exact_acc": float(head_ea.mean()),
                        "per_task_selection": sel},
           "configs": {k: R[k] for k in R},
           "groupkfold5": {"per_fold": cv, "test_mean": cv_mean, "test_std": cv_std},
           "total_seconds": round(time.time() - t0)}
    (PROJ / "results" / "p25_main_table.json").write_text(json.dumps(out, indent=1))
    print(f":  {headline:.4f} CI{ci} | CV {cv_mean:.4f}±{cv_std:.4f} "
          f"|  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
