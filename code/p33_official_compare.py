"""P3.3 baseline: ordinal.ckpt (DINOv2 ViT-S + CORN)  test
 (QWK + linear kappa + exact acc + ECE)

 cxr_dataset.py :
  PNG float32 [1,H,W] -> CenterCrop(448,448,pad_if_needed) -> z-normalization
 results/p33_official_compare.json
"""
import io, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from torchvision import transforms as T

PROJ = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = Path(__import__("os").environ.get(
    "TAIXRAY_OFFICIAL_REPO", PROJ.parent / "taixray-official-repo"))
sys.path.insert(0, str(OFFICIAL_REPO))
DATA = PROJ / "taixray" / "data"
CKPT = PROJ / "official-ckpt" / "ordinal.ckpt"
OFFICIAL_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Left", "PleuralEffusion_Right",
                 "PulmonaryOpacities_Left", "PulmonaryOpacities_Right", "Atelectasis_Left", "Atelectasis_Right"]
OUR_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right", "PleuralEffusion_Left",
            "PulmonaryOpacities_Right", "PulmonaryOpacities_Left", "Atelectasis_Right", "Atelectasis_Left"]
N_LEVELS = {"HeartSize": 4, **{c: 5 for c in OUR_COLS if c != "HeartSize"}}
LIMIT = int(__import__("os").environ.get("P33_LIMIT", "0"))


def eval_transform():
    return T.Compose([T.CenterCrop((448, 448))])


def load_batches(split, bs=48):
    tf = eval_transform()
    for path in sorted(DATA.glob(f"{split}-*.parquet")):
        t = pq.read_table(path, columns=["Image", "UID"] + OUR_COLS)
        imgs = t.column("Image").to_pylist()
        Y = torch.from_numpy(t.select(OUR_COLS).to_pandas().to_numpy(np.int64))
        uids = t.column("UID").to_pylist()
        for s in range(0, len(imgs), bs):
            xs, ys, us = [], [], []
            for i in range(s, min(s + bs, len(imgs))):
                arr = np.asarray(Image.open(io.BytesIO(imgs[i]["bytes"]))).astype(np.float32)
                x = torch.from_numpy(arr)[None]
                x = tf(x)
                x = (x - x.mean()) / x.std().clamp(min=1e-6)
                xs.append(x)
                ys.append(Y[i]); us.append(uids[i])
            yield torch.stack(xs), torch.stack(ys), us


def main():
    from cxr.models.mst import MSTRegression
    print("ckpt...", flush=True)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    print("ckpt keys:", list(sd.keys())[:4], flush=True)
    print("epoch:", ckpt.get("epoch"), "| hp:", str(ckpt.get("hyper_parameters"))[:120], flush=True)
    out_dim = [v.shape for k, v in sd.items() if k.endswith("linear.weight")]
    print("linear.weight:", out_dim, flush=True)
    out_ch = int(out_dim[0][0])
    model = MSTRegression(in_ch=1, out_ch=out_ch, task="ordinal",
                          loss_kwargs={"class_labels_num": [N_LEVELS[c] - 1 for c in OFFICIAL_COLS]})
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("missing:", missing[:5], flush=True)
    model = model.cuda().eval()

    n_done, n_total = 0, 0
    preds = {c: [] for c in OFFICIAL_COLS}
    ys_all = {c: [] for c in OFFICIAL_COLS}
    uids_all = []
    t0 = time.time()
    with torch.no_grad():
        for x, Y, us in load_batches("test"):
            if LIMIT and n_done >= LIMIT:
                break
            logit = model(x.cuda()).float().cpu()
            off = 0
            for c in OFFICIAL_COLS:
                k = N_LEVELS[c] - 1
                blk = logit[:, off:off + k]
                cum = torch.cumprod(torch.sigmoid(blk), dim=1)  # CORN conditionals -> joint cumulative
                preds[c].append((cum > 0.5).sum(1))
                ys_all[c].append(Y[:, OUR_COLS.index(c)])
                off += k
            uids_all += us
            n_done += len(x)
            if n_done % 4800 < 48:
                print(f"  {n_done} ({time.time()-t0:.0f}s)", flush=True)
    print(f" {n_done}  ({time.time()-t0:.0f}s)", flush=True)

    res = {"model": "official ordinal.ckpt (DINOv2 ViT-S/14 + linear CORN head, 448 center-crop, per-image z-norm)",
           "n_eval": n_done, "per_task": {}}
    for c in OFFICIAL_COLS:
        p = torch.cat(preds[c]).numpy()[:n_done]
        y = torch.cat(ys_all[c]).numpy()[:n_done]
        k = N_LEVELS[c]
        res["per_task"][c] = {
            "qwk": round(float(cohen_kappa_score(y, p, weights="quadratic", labels=list(range(k)))), 4),
            "linear_kappa": round(float(cohen_kappa_score(y, p, weights="linear", labels=list(range(k)))), 4),
            "exact_acc": round(float((y == p).mean()), 4),
        }
        print(f"  {c:26s} QWK {res['per_task'][c]['qwk']:.3f} | linK {res['per_task'][c]['linear_kappa']:.3f} "
              f"| acc {res['per_task'][c]['exact_acc']:.3f}", flush=True)
    for m in ["qwk", "linear_kappa", "exact_acc"]:
        res[f"mean_{m}"] = round(float(np.mean([res["per_task"][c][m] for c in OFFICIAL_COLS])), 4)
    print(f": QWK {res['mean_qwk']} | linearK {res['mean_linear_kappa']} | acc {res['mean_exact_acc']}", flush=True)
    (PROJ / "results" / "p33_official_compare.json").write_text(json.dumps(res, indent=1))
    print(" results/p33_official_compare.json", flush=True)


if __name__ == "__main__":
    main()
