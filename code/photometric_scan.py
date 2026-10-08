""" 215,381

 results/photometric_scan.parquet:  UID/ spine_ratio / corner_ratio / frac_high
:  (spine_ratio>1);
"""
import io, glob, sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "photometric_scan.parquet"


def scan_shard(path_str):
    path = Path(path_str)
    t = pq.read_table(path, columns=["Image", "UID"])
    col = t.column("Image")
    rows = []
    for i in range(len(col)):
        try:
            a = np.asarray(Image.open(io.BytesIO(col[i].as_py()["bytes"]))).astype(np.float32)
        except Exception:
            continue
        rng = a.max() - a.min()
        if rng <= 0:
            continue
        a = (a - a.min()) / rng
        h, w = a.shape
        corners = np.mean([a[:20, :20].mean(), a[:20, -20:].mean(),
                           a[-20:, :20].mean(), a[-20:, -20:].mean()])
        spine = a[:, max(w // 2 - w // 14, 0):w // 2 + w // 14].mean()
        rows.append((t.column("UID")[i].as_py(), spine / (np.median(a) + 1e-6),
                     corners / (np.median(a) + 1e-6), float((a > 0.75).mean())))
    df = pd.DataFrame(rows, columns=["UID", "spine_ratio", "corner_ratio", "frac_high"])
    df["shard"] = path.stem
    return df


if __name__ == "__main__":
    shards = sorted(glob.glob(str(ROOT / "taixray" / "data" / "*.parquet")))
    print(f" {len(shards)} ...")
    with Pool(14) as p:
        dfs = p.map(scan_shard, shards)
    df = pd.concat(dfs, ignore_index=True)
    OUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT)
    n = len(df)
    print(f": {n}  -> {OUT}")
    for col in ["spine_ratio", "corner_ratio", "frac_high"]:
        print(f"{col}: [1,5,25,50,75,95,99] = {np.percentile(df[col], [1,5,25,50,75,95,99]).round(3)}")
    for thr in [0.7, 0.8, 0.9, 1.0]:
        m = df["spine_ratio"] < thr
        print(f"spine_ratio<{thr}: {m.mean()*100:.2f}% (n={m.sum()}),  frac_high  {df.loc[m,'frac_high'].median():.3f}")
