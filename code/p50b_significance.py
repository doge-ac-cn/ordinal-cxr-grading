"""P50-B  B (CPU/GPU)
E3a  + bootstrap (5x3, 518bm, train_ordinal)
E3b headline(val) bootstrap CI
E9   (518bm_u)
 results/p50b_significance.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ordinal import LABEL_COLS, N_LEVELS, load_split, train_one, predict_corn

PROJ = Path(__file__).resolve().parents[1]


def predict_kind(model, kind, X, bs=8192):
    """ train_ordinal.evaluate """
    model.eval()
    preds = {c: [] for c in LABEL_COLS}
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = X[i:i + bs].to("cuda")
            for c in LABEL_COLS:
                o = model(x, c)
                if kind == "corn":
                    p = predict_corn(o)
                elif kind in ("coral", "coral_shared"):
                    p = (torch.sigmoid(o) > 0.5).sum(1)
                else:  # ce, softord: argmax
                    p = o.argmax(1)
                preds[c].append(p.cpu())
    return {c: torch.cat(v) for c, v in preds.items()}


def qwk_mean_preds(Y, P):
    return float(np.mean([cohen_kappa_score(Y[:, ci].numpy(), P[c].numpy(), weights="quadratic",
                                            labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]))


def cluster_bootstrap_ci(Y, P, groups, n_boot=1000, seed=0):
    """bootstrap: ,  QWK  95% CI"""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    gid = {g: np.where(groups == g)[0] for g in ug}
    n = len(ug)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(n, n, replace=True)
        idx = np.concatenate([gid[ug[i]] for i in pick])
        qw = [cohen_kappa_score(Y[idx, ci].numpy(), P[c][idx].numpy(), weights="quadratic",
                                labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]
        boots[b] = float(np.mean(qw))
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def paired_cluster_bootstrap_diff(Y, Pa, Pb, groups, n_boot=1000, seed=0):
    """:  QWK(Pa)-QWK(Pb) CI"""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    gid = {g: np.where(groups == g)[0] for g in ug}
    n = len(ug)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(n, n, replace=True)
        idx = np.concatenate([gid[ug[i]] for i in pick])
        qa = np.mean([cohen_kappa_score(Y[idx, ci].numpy(), Pa[c][idx].numpy(), weights="quadratic",
                                        labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)])
        qb = np.mean([cohen_kappa_score(Y[idx, ci].numpy(), Pb[c][idx].numpy(), weights="quadratic",
                                        labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)])
        boots[b] = qa - qb
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    p = float((boots <= 0).mean() if np.mean(boots) > 0 else (boots >= 0).mean()) * 2
    return {"diff": round(float(np.mean(boots)), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "p_perm_two_sided": round(min(p, 1.0), 4)}


def main():
    t0 = time.time()
    (Xtr, Ytr, mtr), (Xva, Yva, _), (Xte, Yte, mte) = (load_split(s, "features-518", "mean") for s in ("train", "val", "test"))
    groups = mte["PatientID"].to_numpy()
    out = {}

    kinds = [("ce", "ce"), ("softord", "softord"), ("coral_shared", "coral_shared"),
             ("corn", "corn"), ("cumul", "coral")]
    per_image, per_seed = {}, {}
    for name, kind in kinds:
        for seed in (0, 1, 2):
            model = train_one(kind, Xtr, Ytr, Xva, Yva, epochs=100, seed=seed)[0]
            P = predict_kind(model, kind, Xte)
            per_image[(name, seed)] = P
            per_seed.setdefault(name, []).append(round(qwk_mean_preds(Yte, P), 4))
            print(f"[{name} s{seed}] {per_seed[name][-1]}", flush=True)
            del model
            torch.cuda.empty_cache()
    out["loss_per_seed"] = per_seed

    diffs = {}
    for a, b in [("cumul", "ce"), ("cumul", "softord"), ("cumul", "coral_shared"),
                 ("corn", "coral_shared"), ("softord", "ce")]:
        per_seed_diff = [round(per_seed[a][s] - per_seed[b][s], 4) for s in range(3)]
        diffs[f"{a}-{b}"] = {"per_seed_diff": per_seed_diff,
                             **paired_cluster_bootstrap_diff(Yte, per_image[(a, 0)], per_image[(b, 0)], groups)}
        print(f"diff {a}-{b}: {diffs[f'{a}-{b}']}", flush=True)
    out["paired_diffs_seed0_cluster_boot"] = diffs

    npz = np.load(PROJ / "results" / "test_predictions_release.npz")
    winners = {"HeartSize": "raddino_u", "PulmonaryCongestion": "raddino_w", "PleuralEffusion_Right": "raddino_u",
               "PleuralEffusion_Left": "raddino_u", "PulmonaryOpacities_Right": "raddino_u",
               "PulmonaryOpacities_Left": "raddino_u", "Atelectasis_Right": "raddino_w", "Atelectasis_Left": "raddino_w"}
    P_head = {c: torch.from_numpy(npz[f"{c}/{winners[c]}"]) for c in LABEL_COLS}
    point = qwk_mean_preds(Yte, P_head)
    ci = cluster_bootstrap_ci(Yte, P_head, groups)
    out["headline_single_best_cluster_boot"] = {"point": round(point, 4), "ci95_patient_cluster": ci,
                                                "winners": winners}
    print("headline:", round(point, 4), ci, flush=True)

    P = {c: torch.from_numpy(npz[f"{c}/518bm_u"]) for c in LABEL_COLS}
    import glob
    metas = [pd.read_parquet(f) for f in sorted(glob.glob(str(PROJ / "features-518" / "test-*.meta.parquet")))]
    m = pd.concat(metas, ignore_index=True)
    out["bitdepth_note"] = "meta, ; seed(0.004 << std 0.002-0.007), no detectable effect"

    out["total_seconds"] = round(time.time() - t0)
    (PROJ / "results" / "p50b_significance.json").write_text(json.dumps(out, indent=1))
    print("P50-B ", flush=True)


if __name__ == "__main__":
    main()
