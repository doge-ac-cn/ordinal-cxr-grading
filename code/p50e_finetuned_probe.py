"""P50-E  E (A M5 ):
(p44 v2) +
 ~0.60 => ();  ~0.43 =>
 results/p50e_finetuned_probe.json
"""
import io
import json
import sys
import time
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
BS, WORKERS = 64, 10


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


def extract_split(bb, split, invert_uids, out_dir):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class DS(Dataset):
        def __init__(self, path):
            t = pq.read_table(path, columns=["Image", "UID"])
            self.imgs = t.column("Image").to_pylist()
            self.uids = t.column("UID").to_pylist()
            self.meta = pq.read_table(path).to_pandas()

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            return decode(self.imgs[i]["bytes"], self.uids[i] in invert_uids), i

    feats_all = []
    n_total = 0
    t0 = time.time()
    for path in sorted(DATA.glob(f"{split}-*.parquet")):
        out_npz = out_dir / f"{path.stem}.npz"
        if out_npz.exists():
            d = np.load(out_npz)
            feats_all.append(d["f"])
            n_total += len(d["f"])
            continue
        ds = DS(path)
        dl = DataLoader(ds, batch_size=BS, num_workers=WORKERS)
        feats = np.zeros((len(ds), 768), dtype=np.float16)
        with torch.autocast("cuda", dtype=torch.float16):
            for x, idx in dl:
                with torch.no_grad():
                    xb = x.cuda().float().permute(0, 3, 1, 2) / 255.
                    xb = (xb - MEAN.cuda()) / STD.cuda()
                    tok = bb(pixel_values=xb, interpolate_pos_encoding=True).last_hidden_state.float()
                feats[idx.numpy()] = tok[:, 1:].mean(1).cpu().numpy().astype(np.float16)
        np.savez_compressed(out_npz, f=feats)
        feats_all.append(feats)
        n_total += len(ds)
        print(f"[{split}] {path.stem}:  {n_total} ({time.time()-t0:.0f}s)", flush=True)
    return np.concatenate(feats_all)


def main():
    import torch.nn as nn
    from transformers import Dinov2Model
    from sklearn.metrics import cohen_kappa_score

    out_dir = PROJ / "features-ftb"
    out_dir.mkdir(exist_ok=True)
    scan = pd.read_parquet(PROJ / "results" / "photometric_scan.parquet")
    invert_uids = set(scan.loc[(scan.spine_ratio < 0.9) & (scan.frac_high > 0.15), "UID"])

    bb = Dinov2Model.from_pretrained("facebook/dinov2-base").cuda().eval()
    ck = torch.load(PROJ / "results" / "p44_best_ckpt.pt", map_location="cpu", weights_only=False)
    bb.load_state_dict(ck["bb"])
    for p in bb.parameters():
        p.requires_grad_(False)
    print(" (val=%.4f)" % ck["val_qwk"], flush=True)

    splits = {}
    for s in ("train", "val", "test"):
        X = extract_split(bb, s, invert_uids, out_dir)
        m = pd.concat([pq.read_table(f).to_pandas() for f in sorted(DATA.glob(f"{s}-*.parquet"))],
                      ignore_index=True)
        assert len(m) == len(X), (s, len(m), len(X))
        splits[s] = (torch.from_numpy(X.astype(np.float32)),
                     torch.from_numpy(m[LABEL_COLS].to_numpy(np.int64)))
        print(f"{s}: {X.shape}", flush=True)

    class CumHead(nn.Module):
        def __init__(self, dim, cols):
            super().__init__()
            self.cols = cols
            self.heads = nn.ModuleDict({c: nn.Linear(dim, N_LEVELS[c] - 1) for c in cols})

        def forward(self, x, col):
            return self.heads[col](x)

    Xtr, Ytr = splits["train"]
    Xva, Yva = splits["val"]
    Xte, Yte = splits["test"]
    per = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        model = CumHead(768, LABEL_COLS).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        best, bad, best_state = -1, 0, None

        def qwk_mean(model, X, Y):
            model.eval()
            preds = {}
            with torch.no_grad():
                for c in LABEL_COLS:
                    z = torch.cat([model(X[i:i + 16384].cuda(), c) for i in range(0, len(X), 16384)]).cpu()
                    preds[c] = (torch.sigmoid(z) > 0.5).sum(1)
            return float(np.mean([cohen_kappa_score(Y[:, ci].numpy(), preds[c].numpy(), weights="quadratic",
                                                    labels=list(range(N_LEVELS[c]))) for ci, c in enumerate(LABEL_COLS)]))
        for ep in range(100):
            model.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 4096):
                idx = perm[i:i + 4096]
                x = Xtr[idx].cuda()
                loss = 0.0
                for ci, c in enumerate(LABEL_COLS):
                    y = Ytr[idx, ci].cuda()
                    t = (y.unsqueeze(1) > torch.arange(N_LEVELS[c] - 1, device=y.device).unsqueeze(0)).float()
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(model(x, c), t)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            mq = qwk_mean(model, Xva, Yva)
            if mq > best:
                best, bad = mq, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    break
        model.load_state_dict(best_state)
        per.append(round(qwk_mean(model, Xte, Yte), 4))
        print(f"seed{seed}: test {per[-1]}", flush=True)
    out = {"config": "linear cumulative head on FROZEN fine-tuned (p44 v2) backbone features @224 mean-pool, 3 seeds",
           "per_seed": per, "mean": round(float(np.mean(per)), 4),
           "anchors": {"frozen generic probe": 0.4130, "full fine-tune": 0.6044,
                       "label-space distillation student": 0.4227}}
    (PROJ / "results" / "p50e_finetuned_probe.json").write_text(json.dumps(out, indent=1))
    print("P50-E :", out["mean"], flush=True)


if __name__ == "__main__":
    main()
