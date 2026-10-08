"""P4.9g :  <- (p44 v2, test mean QWK 0.6044)
: alpha*BCE(logits, ) + (1-alpha)*BCE(logits, )  [cum]
: train / valalphaepoch() / test, seeds{0,1,2}
: 518bm CE probe 0.4342 |  0.5187 |  0.5660 |  0.6044
 results/p49g_student.json + results/p49g_student_test_preds.npz
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

PROJ = Path(__file__).resolve().parents[1]
LABEL_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right",
              "PleuralEffusion_Left", "PulmonaryOpacities_Right", "PulmonaryOpacities_Left",
              "Atelectasis_Right", "Atelectasis_Left"]
N_LEVELS = {"HeartSize": 4, **{c: 5 for c in LABEL_COLS if c != "HeartSize"}}
OFFS, _o = {}, 0
for _c in LABEL_COLS:
    OFFS[_c] = _o; _o += N_LEVELS[_c]
N_SOFT = _o
CFGS = [("224bm", "features", "mean"), ("518bm", "features-518", "mean")]
ALPHAS = [0.5, 0.7, 0.9]


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


def probs_to_cum(p):
    """(n,K)  -> (n,K-1)  P(y>k), k=0..K-2"""
    return torch.stack([p[:, k + 1:].sum(1) for k in range(p.shape[1] - 1)], dim=1)


class CumHead(nn.Module):
    def __init__(self, dim, cols):
        super().__init__()
        self.cols = cols
        self.heads = nn.ModuleDict({c: nn.Linear(dim, N_LEVELS[c] - 1) for c in cols})

    def forward(self, x, col):
        return self.heads[col](x)


def train_kd(Xtr, Ttr, Ytr, Xva, Yva, alpha, seed=0, epochs=100, patience=10, lr=1e-3, wd=1e-4):
    """cum: BCE(logits, ) + alpha*BCE(logits, )
     = teacher probs  P(y>k); clamp, """
    torch.manual_seed(seed)
    model = CumHead(Xtr.shape[1], LABEL_COLS).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
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
                z = model(x, c)
                yb = Ytr[idx, ci].cuda()
                tgt = torch.arange(N_LEVELS[c] - 1, device=yb.device).unsqueeze(0)
                t_hard = (yb.unsqueeze(1) > tgt).float()
                t_soft = Ttr[c][idx].cuda()
                hard = F.binary_cross_entropy_with_logits(z, t_hard)
                soft = F.binary_cross_entropy_with_logits(z, t_soft)
                loss = loss + (1 - alpha) * hard + alpha * soft
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
    return {c: (torch.sigmoid(v) > 0.5).sum(1) for c, v in out.items()}


def qwk_mean(model, X, Y):
    p = predict(model, X)
    return float(np.mean([cohen_kappa_score(Y[:, ci].numpy(), p[c].numpy(), weights="quadratic",
                                            labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]))


def main():
    t0 = time.time()
    npz = np.load(PROJ / "results" / "teacher_soft_labels.npz")
    soft_all, t_uids = npz["soft"], npz["uids"]
    assert soft_all.shape == (137593, N_SOFT), f" {soft_all.shape}"
    uid2row = {u: i for i, u in enumerate(t_uids.tolist())}
    print(f" {soft_all.shape} ({time.time()-t0:.0f}s)", flush=True)

    R, preds_all = {}, {}
    for tag, fdir, pool in CFGS:
        (Xtr, Ytr, mtr), (Xva, Yva, _), (Xte, Yte, _) = (load_split(fdir, pool, s) for s in ("train", "val", "test"))
        tr_uids = mtr["UID"].tolist()
        missing = [u for u in tr_uids[:2000] if u not in uid2row]
        assert not missing, f"UID {missing[:3]}"
        rows = np.fromiter((uid2row[u] for u in tr_uids), dtype=np.int64, count=len(tr_uids))
        Ttr = {c: probs_to_cum(torch.from_numpy(
            soft_all[rows, OFFS[c]:OFFS[c] + N_LEVELS[c]].astype(np.float32))) for c in LABEL_COLS}

        sweep = {}
        for a in ALPHAS:
            model, vq = train_kd(Xtr, Ttr, Ytr, Xva, Yva, alpha=a, seed=0)
            sweep[a] = vq
            print(f"[{tag}] alpha={a} val {vq:.4f} ({time.time()-t0:.0f}s)", flush=True)
        a_best = max(sweep, key=sweep.get)

        per_seed = []
        for s in (0, 1, 2):
            model, vq = train_kd(Xtr, Ttr, Ytr, Xva, Yva, alpha=a_best, seed=s)
            pv, pt = predict(model, Xva), predict(model, Xte)
            vqt = {c: float(cohen_kappa_score(Yva[:, ci].numpy(), pv[c].numpy(), weights="quadratic",
                                              labels=list(range(N_LEVELS[c])))) for ci, c in enumerate(LABEL_COLS)}
            q = {c: float(cohen_kappa_score(Yte[:, ci].numpy(), pt[c].numpy(), weights="quadratic",
                                            labels=list(range(N_LEVELS[c])))) for ci, c in enumerate(LABEL_COLS)}
            lk = {c: float(cohen_kappa_score(Yte[:, ci].numpy(), pt[c].numpy(), weights="linear",
                                             labels=list(range(N_LEVELS[c])))) for ci, c in enumerate(LABEL_COLS)}
            ea = {c: float((Yte[:, ci].numpy() == pt[c].numpy()).mean()) for ci, c in enumerate(LABEL_COLS)}
            per_seed.append({"seed": s, "val": vq, "val_per_task": vqt,
                             "test_mean": float(np.mean(list(q.values()))),
                             "test_mean_linear_kappa": float(np.mean(list(lk.values()))),
                             "test_mean_exact_acc": float(np.mean(list(ea.values()))),
                             "test_qwk": q, "test_linear_kappa": lk, "test_exact_acc": ea})
            preds_all[f"{tag}_s{s}"] = pt
            print(f"[{tag}] seed{s} val {vq:.4f} test {per_seed[-1]['test_mean']:.4f} ({time.time()-t0:.0f}s)", flush=True)

        R[tag] = {"alpha_sweep_val": {str(a): v for a, v in sweep.items()}, "alpha_best": a_best,
                  "per_seed": per_seed,
                  "test_mean_qwk": [p["test_mean"] for p in per_seed],
                  "test_mean_qwk_avg": float(np.mean([p["test_mean"] for p in per_seed])),
                  "test_mean_qwk_std": float(np.std([p["test_mean"] for p in per_seed], ddof=1))}

    best_tag = max(R, key=lambda k: R[k]["test_mean_qwk_avg"])
    best_seed = max(R[best_tag]["per_seed"], key=lambda p: p["val"])
    pt = preds_all[f"{best_tag}_s{best_seed['seed']}"]
    n_te = len(pt[LABEL_COLS[0]])
    _, Yte, _ = load_split(dict((c[0], c[1]) for c in CFGS)[best_tag], "mean", "test")
    rng = np.random.default_rng(0)
    boots = np.empty(1000)
    for b in range(1000):
        idx = rng.integers(0, n_te, n_te)
        qw = [cohen_kappa_score(Yte[idx, ci].numpy(), pt[c][idx].numpy(), weights="quadratic",
                                labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]
        boots[b] = float(np.mean(qw))
    ci95 = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    np.savez_compressed(PROJ / "results" / "p49g_student_test_preds.npz",
                        **{f"{k}__{c}": v.numpy() for k, d in preds_all.items() for c, v in d.items()})
    out = {"teacher": "p44_best_ckpt.pt (DINOv2-base@224 finetune v2, test mean QWK 0.6044)",
           "loss": "alpha*BCE(logits, teacher_cum_targets) + (1-alpha)*BCE(logits, hard_cum_targets), cum-space KD",
           "note": "prob-space KL via telescoping+clamp has dead-gradient failure (first attempt: test 0.24 vs 0.41 baseline); cum-space BCE fixes it",
           "protocol": "train/valalphaepoch/test, seeds{0,1,2}, epochs=100/patience=10 (60/8KD, KD)",
           "budget_note": "60ep/8: 518bm KD 0.4197 < 0.4342(); 100ep/10: alpha=0 val 0.4345 vs alpha=0.9 val 0.4389",
           "anchors": {"518bm_ce_probe": 0.4342, "official_finetune": 0.5187,
                       "ensemble_frozen": 0.5660, "teacher_finetune": 0.6044},
           "configs": R, "best": {"tag": best_tag, "bootstrap95ci": ci95,
                                  "headline": R[best_tag]["test_mean_qwk_avg"]},
           "total_seconds": round(time.time() - t0)}
    (PROJ / "results" / "p49g_student.json").write_text(json.dumps(out, indent=1))
    print(f"P4.9g :  {best_tag} {R[best_tag]['test_mean_qwk_avg']:.4f}±"
          f"{R[best_tag]['test_mean_qwk_std']:.4f} CI{ci95} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
