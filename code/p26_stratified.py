"""P2.6 : (518bm cumul)test

: ① (spine<0.9&frac_high>0.15 =  vs ) ② (12bit vs 16bit)
     ③ (+)
 results/p26_stratified.json
"""
import io, json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import cohen_kappa_score

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "taixray" / "data"
FEAT = PROJ / "features-518"
LABEL_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right",
              "PleuralEffusion_Left", "PulmonaryOpacities_Right", "PulmonaryOpacities_Left",
              "Atelectasis_Right", "Atelectasis_Left"]
N_LEVELS = {"HeartSize": 4, **{c: 5 for c in LABEL_COLS if c != "HeartSize"}}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_split(split):
    fs, metas = [], []
    for f in sorted(FEAT.glob(f"{split}-*.npz")):
        fs.append(np.load(f)["feats_mean"])
        metas.append(pd.read_parquet(FEAT / f"{f.stem}.meta.parquet"))
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


def train_cumul(Xtr, Ytr, Xva, Yva, seed=0, epochs=100, patience=10):
    torch.manual_seed(seed)
    model = CumHead(Xtr.shape[1], LABEL_COLS).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    best, bad, best_state = -1, 0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            x = Xtr[idx].cuda()
            loss = sum(bce(model(x, c), cumulative_targets(Ytr[idx, ci].cuda(), N_LEVELS[c]))
                       for ci, c in enumerate(LABEL_COLS))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = {c: torch.zeros(len(Xva), N_LEVELS[c] - 1) for c in LABEL_COLS}
            for j in range(0, len(Xva), 16384):
                x = Xva[j:j + 16384].cuda()
                for ci, c in enumerate(LABEL_COLS):
                    vp[c][j:j + len(x)] = model(x, c).cpu()
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
def predict(model, X, bs=16384):
    model.eval()
    out = {c: torch.zeros(len(X), N_LEVELS[c] - 1) for c in LABEL_COLS}
    for i in range(0, len(X), bs):
        x = X[i:i + bs].cuda()
        for c in LABEL_COLS:
            out[c][i:i + len(x)] = model(x, c).cpu()
    return {c: (torch.sigmoid(v) > 0.5).sum(1).numpy() for c, v in out.items()}


def shard_bitdepth(path_str):
    path = Path(path_str)
    t = pq.read_table(path, columns=["Image", "UID"])
    col = t.column("Image")
    rows = []
    for i in range(len(col)):
        try:
            a = np.asarray(Image.open(io.BytesIO(col[i].as_py()["bytes"])))
            rows.append((t.column("UID")[i].as_py(), int(a.max()) > 4095))
        except Exception:
            continue
    return rows


def main():
    t0 = time.time()
    print("518bm cumul (seed0)...", flush=True)
    Xtr, Ytr, _ = load_split("train")
    Xva, Yva, _ = load_split("val")
    Xte, Yte, mte = load_split("test")
    model, vq = train_cumul(Xtr, Ytr, Xva, Yva)
    preds = predict(model, Xte)
    print(f"val {vq:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print("test...", flush=True)
    shards = sorted(DATA.glob("test-*.parquet"))
    with Pool(12) as pool:
        rows = pool.map(shard_bitdepth, [str(p) for p in shards])
    bit = pd.DataFrame([r for chunk in rows for r in chunk], columns=["UID", "bit16"])
    print(f": 16bit {bit.bit16.mean()*100:.1f}%", flush=True)

    scan = pd.read_parquet(PROJ / "results" / "photometric_scan.parquet")
    m = mte.merge(scan[["UID", "spine_ratio", "frac_high"]], on="UID", how="left") \
           .merge(bit, on="UID", how="left")
    m["inv_group"] = ((m.spine_ratio < 0.9) & (m.frac_high > 0.15)).map({True: "inverted_corrected", False: "normal"})
    m["bit_group"] = m["bit16"].map({True: "16bit_scale", False: "12bit_scale"})

    out = {"model": "518-base-mean cumul seed0", "val_mean_qwk": round(vq, 4)}
    for factor in ["inv_group", "bit_group"]:
        tab = {}
        for g, sub in m.groupby(factor):
            idx = sub.index.to_numpy()
            qs = {c: round(float(cohen_kappa_score(Yte[idx, ci].numpy(), preds[c][idx],
                                                   weights="quadratic", labels=list(range(N_LEVELS[c])))), 4)
                  for ci, c in enumerate(LABEL_COLS)}
            tab[g] = {"n": int(len(idx)), "qwk": qs,
                      "mean_qwk": round(float(np.mean(list(qs.values()))), 4)}
        out[factor] = tab

    per_grade = {}
    for ci, c in enumerate(LABEL_COLS):
        rec = {}
        for g in range(N_LEVELS[c]):
            gid = np.where(Yte[:, ci].numpy() == g)[0]
            rec[str(g)] = {"support": int(len(gid)),
                           "recall": round(float((preds[c][gid] == g).mean()), 3) if len(gid) else None}
        per_grade[c] = rec
    out["per_grade_recall"] = per_grade

    (PROJ / "results" / "p26_stratified.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ["inv_group", "bit_group"]}, indent=1)[:900], flush=True)
    print(f" ({time.time()-t0:.0f}s) -> results/p26_stratified.json", flush=True)


if __name__ == "__main__":
    import time
    main()
