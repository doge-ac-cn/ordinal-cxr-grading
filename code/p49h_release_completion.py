"""P4.8 npz: vitb16/rn50/biomedclip test (p49c)
Table 1  16  test_predictions_release.npz
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p25_main_table import LABEL_COLS, N_LEVELS, load_split, train_cumul, predict

PROJ = Path(__file__).resolve().parents[1]
NPZ = PROJ / "results" / "test_predictions_release.npz"
ADDS = [("vitb16_cls_u", "features-vitb16", "cls", False),
        ("vitb16_cls_w", "features-vitb16", "cls", True),
        ("rn50_avgpool_u", "features-rn50", "cls", False),
        ("rn50_avgpool_w", "features-rn50", "cls", True),
        ("biomedclip_cls_u", "features-biomedclip", "cls", False),
        ("biomedclip_cls_w", "features-biomedclip", "cls", True)]

old = dict(np.load(NPZ))
for tag, fdir, pool, w in ADDS:
    (Xtr, Ytr, _), (Xva, Yva, _), (Xte, Yte, _) = (load_split(fdir, pool, s) for s in ("train", "val", "test"))
    model, vq = train_cumul(Xtr, Ytr, Xva, Yva, weighted=w)
    pt = predict(model, Xte)
    qw = [cohen_kappa_score(Yte[:, ci].numpy(), pt[c].numpy(), weights="quadratic",
                            labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]
    print(f"[{tag}] val {vq:.4f} test {float(np.mean(qw)):.4f}", flush=True)
    for c in LABEL_COLS:
        old[f"{c}/{tag}"] = pt[c].numpy().astype(np.int64)
    del model, Xtr, Xva, Xte
    torch.cuda.empty_cache()

np.savez_compressed(NPZ, **old)
cfgs = sorted(set(k.split("/")[1] for k in old if "/" in k and "y_true" not in k))
print(f"npz: {len(cfgs)} -> {cfgs}", flush=True)
