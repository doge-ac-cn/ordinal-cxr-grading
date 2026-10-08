"""P50-D  D: conformal  +  +  (C M2/M6, B-M7)
 p41  (518bm cumul seed0, val, alpha=0.1)
 results/p50d_conformal_detail.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p41_conformal_proper as cf
from p41_conformal_proper import ALPHA, LABEL_COLS, N_LEVELS, CumHead, CeHead, train_head, class_probs, cif_mass, cif_sets, load_split

PROJ = Path(__file__).resolve().parents[1]


def main():
    print("518...", flush=True)
    Xtr, Ytr = load_split("train")
    Xva, Yva = load_split("val")
    Xte, Yte = load_split("test")
    model = train_head(CumHead, Xtr, Ytr)
    pt = class_probs(model, "cumul", Xte)
    pv = class_probs(model, "cumul", Xva)

    out = {"protocol": "same as p41 (518-base-mean seed0, val calibration, alpha=0.1, CIF interval score)",
           "findings": {}}
    for ci, c in enumerate(LABEL_COLS):
        k = N_LEVELS[c]
        yv, yt = Yva[:, ci].numpy(), Yte[:, ci].numpy()
        sv = cif_mass(pv[c].numpy(), yv)
        n = len(sv)
        q_idx = min(int(np.ceil((n + 1) * (1 - ALPHA))) - 1, n - 1)
        q = float(np.sort(sv)[q_idx])
        lt, rt, sizes = cif_sets(pt[c].numpy(), min(q, 1.0))
        widths = (rt - lt + 1).astype(int)
        cover_by_grade = {}
        for g in range(k):
            m = yt == g
            if m.sum():
                cover_by_grade[g] = {"n": int(m.sum()),
                                     "coverage": round(float(np.mean((yt[m] >= lt[m]) & (yt[m] <= rt[m]))), 4),
                                     "mean_width": round(float(widths[m].mean()), 3)}
        out["findings"][c] = {
            "calibration_n": int(n),
            "q": round(min(q, 1.0), 4),
            "coverage_overall": round(float(np.mean((yt >= lt) & (yt <= rt))), 4),
            "coverage_top_grade": cover_by_grade.get(k - 1),
            "width": {"mean": round(float(widths.mean()), 3), "median": int(np.median(widths)),
                      "iqr": [int(np.percentile(widths, 25)), int(np.percentile(widths, 75))],
                      "frac_ge4": round(float((widths >= 4).mean()), 4),
                      "frac_eq_max": round(float((widths == k).mean()), 4)},
            "coverage_by_true_grade": cover_by_grade}
        print(f"{c}: n={n} width_mean={widths.mean():.2f} frac>=4={out['findings'][c]['width']['frac_ge4']}", flush=True)
    (PROJ / "results" / "p50d_conformal_detail.json").write_text(json.dumps(out, indent=1))
    print("P50-D ", flush=True)


if __name__ == "__main__":
    main()
