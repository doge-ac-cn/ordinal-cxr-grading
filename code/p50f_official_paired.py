"""P5.0f paired official-vs-ours comparison: patient-clustered bootstrap of the mean-QWK
difference between the official checkpoint (chained decoding) and our fine-tuned recipe.
Row alignment is verified by reproducing per-finding QWK from each source.
Writes results/p50f_official_vs_ft_paired.json."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

PROJ = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJ / "code"))
from p50b_significance import paired_cluster_bootstrap_diff  # noqa: E402

LABEL_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right",
              "PleuralEffusion_Left", "PulmonaryOpacities_Right", "PulmonaryOpacities_Left",
              "Atelectasis_Right", "Atelectasis_Left"]
N_LEVELS = {"HeartSize": 4, **{c: 5 for c in LABEL_COLS if c != "HeartSize"}}


def main():
    # test labels + patients from the frozen-feature meta (same row order as all pred files)
    metas = []
    for f in sorted((PROJ / "features-518").glob("test-*.meta.parquet")):
        metas.append(pd.read_parquet(f))
    mte = pd.concat(metas, ignore_index=True)
    Y = {c: mte[c].to_numpy(np.int64) for c in LABEL_COLS}
    groups = mte["PatientID"].to_numpy()
    assert len(mte) == 42928

    off = np.load(PROJ / "results" / "p33_official_preds.npz", allow_pickle=True)
    ft = np.load(PROJ / "results" / "p50c_ft_preds.npz", allow_pickle=True)

    ref_off = json.load(open(PROJ / "results" / "p33_official_compare.json"))["per_task"]
    ref_ft = json.load(open(PROJ / "results" / "finetune_v2_full_metrics.json"))["per_task"]

    po, pf = {}, {}
    for c in LABEL_COLS:
        k = N_LEVELS[c]
        o = off[f"{c}/chained"].astype(np.int64)
        f = ft[c].astype(np.int64)
        qo = float(cohen_kappa_score(Y[c], o, weights="quadratic", labels=list(range(k))))
        qf = float(cohen_kappa_score(Y[c], f, weights="quadratic", labels=list(range(k))))
        assert abs(qo - ref_off[c]["qwk"]) < 5e-4, (c, qo, ref_off[c]["qwk"])
        assert abs(qf - ref_ft[c]["qwk"]) < 5e-4, (c, qf, ref_ft[c]["qwk"])
        po[c], pf[c] = o, f
    print("alignment verified: per-finding QWK reproduces both sources", flush=True)

    out = {"protocol": "paired patient-clustered bootstrap, official chained decoding vs ours-FT (12-ep recipe), 1000 resamples",
           "per_task_point_diff": {}}
    Pt_o = {c: torch.from_numpy(po[c]) for c in LABEL_COLS}
    Pt_f = {c: torch.from_numpy(pf[c]) for c in LABEL_COLS}
    Ymat = torch.from_numpy(np.stack([Y[c] for c in LABEL_COLS], axis=1))
    for c in LABEL_COLS:
        k = N_LEVELS[c]
        d_pt = float(cohen_kappa_score(Y[c], po[c], weights="quadratic", labels=list(range(k)))) - \
               float(cohen_kappa_score(Y[c], pf[c], weights="quadratic", labels=list(range(k))))
        out["per_task_point_diff"][c] = round(d_pt, 4)
    d = paired_cluster_bootstrap_diff(Ymat, Pt_o, Pt_f, groups)
    out["mean_diff_cluster_boot"] = d
    (PROJ / "results" / "p50f_official_vs_ft_paired.json").write_text(json.dumps(out, indent=1))
    print("per-finding diffs:", out["per_task_point_diff"], flush=True)
    print(f"MEAN diff official - oursFT = {d['diff']:+.4f}, 95% patient-cluster CI {d['ci95']}", flush=True)


if __name__ == "__main__":
    main()
