"""checkpoint: QWK + linear kappa + exact acc ()

 results/p23_best_ckpt.pt (bb=Dinov2Model, head=CumHead), test
 results/finetune_full_metrics.json
"""
import io, json, time
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
SCAN = pd.read_parquet(PROJ / "results" / "photometric_scan.parquet")
INVERT_UIDS = set(SCAN.loc[(SCAN.spine_ratio < 0.9) & (SCAN.frac_high > 0.15), "UID"])
LABEL_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right",
              "PleuralEffusion_Left", "PulmonaryOpacities_Right", "PulmonaryOpacities_Left",
              "Atelectasis_Right", "Atelectasis_Left"]
META_COLS = ["UID", "PatientID", "Fold", "Split"] + LABEL_COLS
N_LEVELS = {"HeartSize": 4, **{c: 5 for c in LABEL_COLS if c != "HeartSize"}}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
BS = 48


def decode(img_bytes, invert=False):
    arr = np.asarray(Image.open(io.BytesIO(img_bytes))).astype(np.float32)
    if invert:
        arr = arr.max() - arr
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    if p99 <= p1:
        p1, p99 = arr.min(), max(arr.max(), arr.min() + 1)
    arr = np.clip((arr - p1) / (p99 - p1), 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(arr).convert("RGB").resize((224, 224), Image.BILINEAR))


def iter_batches(split):
    for path in sorted(DATA.glob(f"{split}-*.parquet")):
        t = pq.read_table(path, columns=["Image", "UID"] + LABEL_COLS)
        imgs = t.column("Image").to_pylist()
        inv = [u in INVERT_UIDS for u in t.column("UID").to_pylist()]
        Y = torch.from_numpy(t.select(LABEL_COLS).to_pandas().to_numpy(np.int64))
        for s in range(0, len(imgs), BS):
            xs, ys = [], []
            for i in range(s, min(s + BS, len(imgs))):
                xs.append(decode(imgs[i]["bytes"], inv[i]))
                ys.append(Y[i])
            x = torch.stack([torch.from_numpy(a.transpose(2, 0, 1).astype(np.float32) / 255.) for a in xs])
            yield x, torch.stack(ys)


class CumHead(nn.Module):
    def __init__(self, dim, cols):
        super().__init__()
        self.cols = cols
        self.heads = nn.ModuleDict({c: nn.Linear(dim, N_LEVELS[c] - 1) for c in cols})

    def forward(self, x, col):
        return self.heads[col](x)


def cumulative_targets(y, k):
    t = torch.arange(k - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > t).float()


def main():
    from transformers import Dinov2Model
    t0 = time.time()
    ck = torch.load(PROJ / "results" / "p23_best_ckpt.pt", map_location="cpu", weights_only=False)
    bb = Dinov2Model.from_pretrained("facebook/dinov2-base").cuda().eval()
    bb.load_state_dict(ck["bb"])
    head = CumHead(768, LABEL_COLS).cuda().eval()
    head.load_state_dict(ck["head"])
    print(f"checkpoint (val_qwk={ck['val_qwk']:.4f}) {time.time()-t0:.0f}s", flush=True)

    preds = {c: torch.zeros(42928, dtype=torch.int64) for c in LABEL_COLS}
    ys = {c: torch.zeros(42928, dtype=torch.int64) for c in LABEL_COLS}
    pos = 0
    bce = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, Y in iter_batches("test"):
            x = x.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                tok = bb(pixel_values=(x - MEAN.cuda()) / STD.cuda(),
                         interpolate_pos_encoding=True).last_hidden_state.float()
            feat = tok[:, 1:].mean(1)
            for ci, c in enumerate(LABEL_COLS):
                preds[c][pos:pos + len(x)] = (torch.sigmoid(head.heads[c](feat)) > 0.5).sum(1).cpu()
                ys[c][pos:pos + len(x)] = Y[:, ci]
            pos += len(x)
            if pos % 4800 < BS:
                print(f"  {pos} ({time.time()-t0:.0f}s)", flush=True)

    res = {"model": "DINOv2-base@224 finetune (best ckpt, ep2)", "n": pos, "per_task": {}}
    for ci, c in enumerate(LABEL_COLS):
        k = N_LEVELS[c]
        p, y = preds[c].numpy(), ys[c].numpy()
        res["per_task"][c] = {
            "qwk": round(float(cohen_kappa_score(y, p, weights="quadratic", labels=list(range(k)))), 4),
            "linear_kappa": round(float(cohen_kappa_score(y, p, weights="linear", labels=list(range(k)))), 4),
            "exact_acc": round(float((y == p).mean()), 4),
        }
        print(f"  {c:26s} QWK {res['per_task'][c]['qwk']:.3f} | linK {res['per_task'][c]['linear_kappa']:.3f} "
              f"| acc {res['per_task'][c]['exact_acc']:.3f}", flush=True)
    for m in ["qwk", "linear_kappa", "exact_acc"]:
        res[f"mean_{m}"] = round(float(np.mean([res["per_task"][c][m] for c in LABEL_COLS])), 4)
    print(f": QWK {res['mean_qwk']} | linK {res['mean_linear_kappa']} | acc {res['mean_exact_acc']}", flush=True)
    (PROJ / "results" / "finetune_full_metrics.json").write_text(json.dumps(res, indent=1))
    print(" results/finetune_full_metrics.json", flush=True)


if __name__ == "__main__":
    main()
