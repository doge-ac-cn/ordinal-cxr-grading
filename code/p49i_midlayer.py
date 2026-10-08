"""P4.9e : DINOv2-base layer-6 hidden states, 224, CLS+patch-mean
: mid cumulative probe (u/w × 3 seeds) vs  224bc (0.4103/0.4139)
 results/p49i_midlayer.json
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
from extract_features import ShardDS, norm_batch
from torch.utils.data import DataLoader

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "taixray" / "data"
OUT = PROJ / "features-mid"
OUT.mkdir(exist_ok=True)
RES, LAYER, BATCH = 224, 6, 256
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p25_main_table import LABEL_COLS, N_LEVELS, load_split, train_cumul, predict

def extract(shard_idx=0, num_groups=2):
    from transformers import Dinov2Model
    device = "cuda"
    bb = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
    for p in bb.parameters():
        p.requires_grad_(False)
    scan = pd.read_parquet(PROJ / "results" / "photometric_scan.parquet")
    invert_uids = set(scan.loc[(scan.spine_ratio < 0.9) & (scan.frac_high > 0.15), "UID"])
    shards = []
    for s in ("train", "val", "test"):
        shards += sorted(DATA.glob(f"{s}-*.parquet"))
    mine = [p for n, p in enumerate(shards) if n % num_groups == shard_idx]
    t0, n_done = time.time(), 0
    for path in mine:
        out_npz = OUT / f"{path.stem}.npz"
        if out_npz.exists():
            continue
        ds = ShardDS(path, invert_uids, RES)
        dl = DataLoader(ds, batch_size=BATCH, num_workers=10)
        dim = 768
        feats = np.zeros((len(ds), dim), dtype=np.float16)
        feats_mean = np.zeros((len(ds), dim), dtype=np.float16)
        with torch.autocast("cuda", dtype=torch.float16):
            for idx, x in dl:
                with torch.no_grad():
                    xin = norm_batch(x).to(device, non_blocking=True)
                    out = bb(pixel_values=xin, interpolate_pos_encoding=True, output_hidden_states=True)
                tok = out.hidden_states[LAYER].float()
                feats[idx.numpy()] = tok[:, 0].cpu().numpy().astype(np.float16)
                feats_mean[idx.numpy()] = tok[:, 1:].mean(1).cpu().numpy().astype(np.float16)
        np.savez_compressed(out_npz, feats=feats, feats_mean=feats_mean)
        ds.meta.to_parquet(OUT / f"{path.stem}.meta.parquet")
        n_done += len(ds)
        print(f"[mid proc{shard_idx}] {path.stem}: {len(ds)} | {n_done/(time.time()-t0):.0f} img/s", flush=True)

def probe():
    R = {}
    for tag, pool in (("mid_cls", "cls"), ("mid_mean", "mean")):
        per = []
        for seed in (0, 1, 2):
            (Xtr, Ytr, _), (Xva, Yva, _), (Xte, Yte, _) = (load_split("features-mid", pool, s) for s in ("train", "val", "test"))
            model, vq = train_cumul(Xtr, Ytr, Xva, Yva, seed=seed)
            pt = predict(model, Xte)
            q = [cohen_kappa_score(Yte[:, ci].numpy(), pt[c].numpy(), weights="quadratic",
                                   labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]
            per.append(float(np.mean(q)))
            print(f"[{tag} s{seed}] test {per[-1]:.4f}", flush=True)
            del model, Xtr, Xva, Xte
            torch.cuda.empty_cache()
        R[tag] = {"per_seed": per, "mean": float(np.mean(per)), "std": float(np.std(per, ddof=1))}
    out = {"config": "DINOv2-base layer-6 @224, cumulative head, 3 seeds",
           "anchors": {"224bc_u_final_layer": 0.4103, "224bc_w_final_layer": 0.4139},
           "results": R}
    (PROJ / "results" / "p49i_midlayer.json").write_text(json.dumps(out, indent=1))
    print(f"P4.9e : {json.dumps({k: round(v['mean'],4) for k,v in R.items()})}", flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["extract", "probe"], required=True)
    ap.add_argument("--shard-idx", type=int, default=0)
    a = ap.parse_args()
    extract(a.shard_idx) if a.mode == "extract" else probe()
