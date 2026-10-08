"""P4.9g  (:  -> GPU)
 results/teacher_soft_labels.npz (137593 x 31 float16  + uids)
"""
import io
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "taixray" / "data"
LABEL_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right",
              "PleuralEffusion_Left", "PulmonaryOpacities_Right", "PulmonaryOpacities_Left",
              "Atelectasis_Right", "Atelectasis_Left"]
N_LEVELS = {"HeartSize": 4, **{c: 5 for c in LABEL_COLS if c != "HeartSize"}}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
BS = 256


def decode_shard(path_str, invert_uids):
    """worker:  -> (uids, uint8 NxHxWx3)"""
    path = Path(path_str)
    t = pq.read_table(path, columns=["Image", "UID"])
    imgs = t.column("Image").to_pylist()
    uids = t.column("UID").to_pylist()
    arrs = np.zeros((len(imgs), 224, 224, 3), dtype=np.uint8)
    for i, item in enumerate(imgs):
        arr = np.asarray(Image.open(io.BytesIO(item["bytes"]))).astype(np.float32)
        if uids[i] in invert_uids:
            arr = arr.max() - arr
        p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
        if p99 <= p1:
            p1, p99 = arr.min(), max(arr.max(), arr.min() + 1)
        arr = np.clip((arr - p1) / (p99 - p1), 0, 1)
        a8 = (arr * 255).astype(np.uint8)
        arrs[i] = np.asarray(Image.fromarray(a8).convert("RGB").resize((224, 224), Image.BILINEAR))
    return uids, arrs


class CumHead(torch.nn.Module):
    def __init__(self, dim, cols):
        super().__init__()
        self.cols = cols
        self.heads = torch.nn.ModuleDict({c: torch.nn.Linear(dim, N_LEVELS[c] - 1) for c in cols})

    def forward(self, x, col):
        return self.heads[col](x)


def main():
    from transformers import Dinov2Model
    t0 = time.time()
    scan = pd.read_parquet(PROJ / "results" / "photometric_scan.parquet")
    invert_uids = set(scan.loc[(scan.spine_ratio < 0.9) & (scan.frac_high > 0.15), "UID"])
    shards = sorted(str(p) for p in DATA.glob("train-*.parquet"))
    print(f" {time.time()-t0:.0f}s |  {len(shards)}", flush=True)

    bb = Dinov2Model.from_pretrained("facebook/dinov2-base").cuda().eval()
    head = CumHead(768, LABEL_COLS).cuda().eval()
    ck = torch.load(PROJ / "results" / "p44_best_ckpt.pt", map_location="cpu", weights_only=False)
    bb.load_state_dict(ck["bb"]); head.load_state_dict(ck["head"])

    with Pool(14) as pool:
        results = pool.starmap(decode_shard, [(p, invert_uids) for p in shards])
    uids, X_all = [], []
    for u, a in results:
        uids += u
        X_all.append(a)
    X_all = np.concatenate(X_all, axis=0)  # (137593,224,224,3) uint8 ~20.7GB
    n_total = len(X_all)
    print(f" {n_total} ({time.time()-t0:.0f}s)", flush=True)

    MEANd, STDd = MEAN.cuda(), STD.cuda()
    o = 0
    offs = {}
    for c in LABEL_COLS:
        offs[c] = o; o += N_LEVELS[c]
    soft = np.zeros((n_total, o), dtype=np.float16)
    pos = 0
    with torch.no_grad():
        for s in range(0, n_total, BS):
            xb = torch.from_numpy(X_all[s:s + BS].transpose(0, 3, 1, 2).astype(np.float32) / 255.).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                tok = bb(pixel_values=(xb - MEANd) / STDd,
                         interpolate_pos_encoding=True).last_hidden_state.float()
            feat = tok[:, 1:].mean(1)
            for c in LABEL_COLS:
                cum = torch.sigmoid(head.heads[c](feat))
                ext = torch.cat([torch.ones(len(xb), 1, device=cum.device), cum,
                                 torch.zeros(len(xb), 1, device=cum.device)], dim=1)
                p = (ext[:, :-1] - ext[:, 1:]).clamp(min=0)
                p = p / p.sum(1, keepdim=True).clamp(min=1e-8)
                soft[pos:pos + len(xb), offs[c]:offs[c] + N_LEVELS[c]] = p.cpu().numpy().astype(np.float16)
            pos += len(xb)
            if pos % 12800 < BS:
                print(f"  {pos} ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(PROJ / "results" / "teacher_soft_labels.npz",
                        soft=soft, uids=np.array(uids))
    print(f": {pos} x {o} ({time.time()-t0:.0f}s) -> results/teacher_soft_labels.npz", flush=True)


if __name__ == "__main__":
    main()
