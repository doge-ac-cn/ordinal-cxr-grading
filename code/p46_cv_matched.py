"""P4.6: GroupKFold(5) by PatientID (epochs=100/patience=10, )
 results/p46_cv_matched.json"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from p25_main_table import load_split, train_cumul, LABEL_COLS, N_LEVELS
from sklearn.metrics import cohen_kappa_score

PROJ = Path(__file__).resolve().parents[1]

def qwk_mean(model, X, Y):
    model.eval()
    preds = {}
    with torch.no_grad():
        for i in range(0, len(X), 16384):
            x = X[i:i+16384].cuda()
            for c in LABEL_COLS:
                preds.setdefault(c, []).append((torch.sigmoid(model(x, c)) > 0.5).sum(1).cpu())
    p = {c: torch.cat(v).numpy() for c, v in preds.items()}
    return float(np.mean([cohen_kappa_score(Y[:, ci].numpy(), p[c], weights="quadratic",
                   labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]))

Xtr, Ytr, mtr = load_split("features-518", "mean", "train")
Xva, Yva, mva = load_split("features-518", "mean", "val")
Xte, Yte, _ = load_split("features-518", "mean", "test")
Xall = torch.cat([Xtr, Xva]); Yall = torch.cat([Ytr, Yva])
mall = pd.concat([mtr, mva], ignore_index=True)
pid_map = {p: i for i, p in enumerate(mall["PatientID"].unique())}
groups = mall["PatientID"].map(pid_map).to_numpy()
gkf = GroupKFold(n_splits=5)
cv = []
t0 = time.time()
for fi, (tri, vai) in enumerate(gkf.split(Xall, groups=groups)):
    model, vq = train_cumul(Xall[tri], Yall[tri], Xall[vai], Yall[vai], epochs=100, patience=10)
    mq = qwk_mean(model, Xte, Yte)
    cv.append({"fold": fi, "val_qwk": round(vq, 4), "test_qwk": round(mq, 4)})
    print(f"fold{fi}: val {vq:.4f} test {mq:.4f} ({time.time()-t0:.0f}s)", flush=True)
m = float(np.mean([f["test_qwk"] for f in cv])); sd = float(np.std([f["test_qwk"] for f in cv], ddof=1))
json.dump({"budget": "epochs=100/patience=10 ()", "per_fold": cv,
           "test_mean": m, "test_std": sd}, open(PROJ/"results"/"p46_cv_matched.json","w"), indent=1)
print(f"CV: {m:.4f}±{sd:.4f}", flush=True)
