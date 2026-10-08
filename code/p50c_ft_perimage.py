"""P50-C  C: FTtest (C M2/M5)
- v2 (p44_best_ckpt) test,
- recall()finding MAEMAE
- RAD-DINO FT()
 results/p50c_ft_perimage.json + results/p50c_ft_preds.npz
"""
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from sklearn.metrics import cohen_kappa_score

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "taixray" / "data"
LABEL_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right",
              "PleuralEffusion_Left", "PulmonaryOpacities_Right", "PulmonaryOpacities_Left",
              "Atelectasis_Right", "Atelectasis_Left"]
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


def infer_split(model_heads, shards, invert_uids, tag):
    """model_heads: {col: (bb, head)} bbbb;  {col: preds_np}"""
    from transformers import Dinov2Model
    bb = model_heads["bb"]
    head = model_heads["head"]
    MEANd, STDd = MEAN.cuda(), STD.cuda()
    preds = {c: [] for c in LABEL_COLS}
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for path in shards:
            t = pq.read_table(path, columns=["Image", "UID"])
            imgs = t.column("Image").to_pylist()
            uids = t.column("UID").to_pylist()
            for s in range(0, len(imgs), BS):
                batch = imgs[s:s + BS]
                ub = uids[s:s + BS]
                arr = np.stack([decode(b["bytes"], u in invert_uids) for b, u in zip(batch, ub)])
                xb = torch.from_numpy(arr.transpose(0, 3, 1, 2).astype(np.float32) / 255.).cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    tok = bb(pixel_values=(xb - MEANd) / STDd,
                             interpolate_pos_encoding=True).last_hidden_state.float()
                feat = tok[:, 1:].mean(1)
                for c in LABEL_COLS:
                    cum = torch.sigmoid(head.heads[c](feat))
                    preds[c].append(((cum > 0.5).sum(1)).cpu())
            n_done += len(imgs)
            print(f"[{tag}] {path.stem}: {n_done} ({time.time()-t0:.0f}s)", flush=True)
    return {c: torch.cat(preds[c]).numpy() for c in LABEL_COLS}


def main():
    import torch.nn as nn
    from transformers import Dinov2Model

    class CumHead(nn.Module):
        def __init__(self, dim, cols):
            super().__init__()
            self.cols = cols
            self.heads = nn.ModuleDict({c: nn.Linear(dim, N_LEVELS[c] - 1) for c in cols})

        def forward(self, x, col):
            return self.heads[col](x)

    scan = pd.read_parquet(PROJ / "results" / "photometric_scan.parquet")
    invert_uids = set(scan.loc[(scan.spine_ratio < 0.9) & (scan.frac_high > 0.15), "UID"])
    shards = sorted(DATA.glob("test-*.parquet"))

    bb = Dinov2Model.from_pretrained("facebook/dinov2-base").cuda().eval()
    head = CumHead(768, LABEL_COLS).cuda().eval()
    ck = torch.load(PROJ / "results" / "p44_best_ckpt.pt", map_location="cpu", weights_only=False)
    bb.load_state_dict(ck["bb"])
    head.load_state_dict(ck["head"])
    print("v2 ckpt", flush=True)

    preds = infer_split({"bb": bb, "head": head}, shards, invert_uids, "ft-v2")
    ys = {c: [] for c in LABEL_COLS}
    for path in shards:
        t = pq.read_table(path, columns=LABEL_COLS)
        for c in LABEL_COLS:
            ys[c].extend(t.column(c).to_pylist())
    Y = {c: np.asarray(ys[c], dtype=np.int64) for c in LABEL_COLS}
    for c in LABEL_COLS:
        assert len(Y[c]) == len(preds[c]), (c, len(Y[c]), len(preds[c]))

    out = {"model": "fine-tuned v2 (p44_best_ckpt), test per-image preds", "n": int(len(preds[LABEL_COLS[0]])),
           "per_finding": {}, "top_grade_recall": {}}
    preds_flat = {}
    for c in LABEL_COLS:
        p, y = preds[c], Y[c]
        k = N_LEVELS[c]
        per_grade = {}
        for g in range(k):
            m = y == g
            per_grade[g] = {"support": int(m.sum()),
                            "recall": round(float((p[m] == g).mean()), 4) if m.sum() else None}
        out["per_finding"][c] = {
            "qwk": round(float(cohen_kappa_score(y, p, weights="quadratic", labels=list(range(k)))), 4),
            "mae": round(float(np.mean(np.abs(y - p))), 4),
            "exact": round(float((y == p).mean()), 4),
            "per_grade_recall": per_grade}
        out["top_grade_recall"][c] = per_grade[k - 1]["recall"]
        preds_flat[c] = p
    out["mean_qwk"] = round(float(np.mean([out["per_finding"][c]["qwk"] for c in LABEL_COLS])), 4)
    out["mean_mae"] = round(float(np.mean([out["per_finding"][c]["mae"] for c in LABEL_COLS])), 4)
    np.savez_compressed(PROJ / "results" / "p50c_ft_preds.npz", **preds_flat)
    (PROJ / "results" / "p50c_ft_perimage.json").write_text(json.dumps(out, indent=1))
    print("P50-C ", json.dumps(out["top_grade_recall"]), "meanQWK", out["mean_qwk"], flush=True)


if __name__ == "__main__":
    main()
