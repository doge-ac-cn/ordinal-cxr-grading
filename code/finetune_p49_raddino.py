"""P4.4/P4.9g RAD-DINO:  +  (=v2: +12ep patience3)

(): ->val->test->results->//STATUS->
: mean-pool patch tokens() + BCE() + AdamW(2e-5/1e-3) + 3 epochs
: nohup python3 finetune_p23.py > finetune_3ep.log 2>&1 &  ()
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
LOCK = Path(".lock")
LABEL_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right",
              "PleuralEffusion_Left", "PulmonaryOpacities_Right", "PulmonaryOpacities_Left",
              "Atelectasis_Right", "Atelectasis_Left"]
META_COLS = ["UID", "PatientID", "Fold", "Split"] + LABEL_COLS
N_LEVELS = {"HeartSize": 4, **{c: 5 for c in LABEL_COLS if c != "HeartSize"}}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
EPOCHS, BS, LR_BB, LR_HEAD, WD = 12, 48, 2e-5, 1e-3, 0.05
PATIENCE = 3


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


class ShardIter:
    """: tensor + """
    def __init__(self, split, invert_uids, bs):
        import torchvision.transforms as _T
        self._rot = _T.RandomRotation((-7, 7))
        self._aff = _T.RandomAffine(degrees=0, translate=(0.05, 0.05))
        self.paths = sorted(DATA.glob(f"{split}-*.parquet"))
        self.invert_uids, self.bs = invert_uids, bs

    def __iter__(self, augment=False):
        for path in self.paths:
            t = pq.read_table(path, columns=["Image"] + META_COLS)
            imgs = t.column("Image").to_pylist()
            inv = [u in self.invert_uids for u in t.column("UID").to_pylist()]
            Y = torch.from_numpy(t.select(LABEL_COLS).to_pandas().to_numpy(np.int64))
            buf_x, buf_y = [], []
            for i in range(len(imgs)):
                buf_x.append(decode(imgs[i]["bytes"], inv[i]))
                buf_y.append(Y[i])
                if len(buf_x) == self.bs:
                    yield self.pack(buf_x, buf_y, augment)
                    buf_x, buf_y = [], []
            if buf_x:
                yield self.pack(buf_x, buf_y, augment)

    def pack(self, xs, ys, augment=False):
        x = torch.stack([torch.from_numpy(a.transpose(2, 0, 1).astype(np.float32) / 255.) for a in xs])
        if augment:
            x = self._rot(x)
            x = self._aff(x)
            g = 0.8 + 0.4 * torch.rand(1).item()
            x = x ** g
        return x, torch.stack(ys)

    def __len__(self):
        import pyarrow.parquet as pq2
        return sum(pq2.read_metadata(p).num_rows for p in self.paths)


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


@torch.no_grad()
def evaluate(bb, head, it, n_total):
    bb.eval(); head.eval()
    preds = {c: torch.zeros(n_total, dtype=torch.int64) for c in LABEL_COLS}
    ys = {c: torch.zeros(n_total, dtype=torch.int64) for c in LABEL_COLS}
    pos = 0
    for x, Y in it:
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            tok = bb(pixel_values=(x - MEAN.cuda()) / STD.cuda(),
                     interpolate_pos_encoding=True).last_hidden_state.float()
        feat = tok[:, 1:].mean(1)
        for ci, c in enumerate(LABEL_COLS):
            preds[c][pos:pos + len(x)] = (torch.sigmoid(head.heads[c](feat)) > 0.5).sum(1).cpu()
            ys[c][pos:pos + len(x)] = Y[:, ci]
        pos += len(x)
    bb.train()
    return {c: cohen_kappa_score(ys[c].numpy(), preds[c].numpy(), weights="quadratic",
                                 labels=list(range(N_LEVELS[c]))) for c in LABEL_COLS}


def main():
    from transformers import Dinov2Model
    t0 = time.time()
    scan = pd.read_parquet(PROJ / "results" / "photometric_scan.parquet")
    invert_uids = set(scan.loc[(scan.spine_ratio < 0.9) & (scan.frac_high > 0.15), "UID"])
    device = "cuda"
    bb = Dinov2Model.from_pretrained("microsoft/rad-dino").to(device)
    head = CumHead(768, LABEL_COLS).to(device)
    print(f"[P2.3]  {time.time()-t0:.0f}s", flush=True)

    train_it = ShardIter("train", invert_uids, BS)
    val_it = ShardIter("val", invert_uids, 128)
    test_it = ShardIter("test", invert_uids, 128)
    n_tr, n_va, n_te = len(train_it), len(val_it), len(test_it)
    print(f"[P2.3] train {n_tr} val {n_va} test {n_te}", flush=True)

    opt = torch.optim.AdamW([{"params": bb.parameters(), "lr": LR_BB},
                             {"params": head.parameters(), "lr": LR_HEAD}], weight_decay=WD)
    import pyarrow.parquet as _pq
    _rows = [_pq.read_metadata(_p).num_rows for _p in sorted(DATA.glob("train-*.parquet"))]
    steps_per_epoch = sum((r + BS - 1) // BS for r in _rows)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_BB, LR_HEAD], total_steps=steps_per_epoch * EPOCHS, pct_start=0.06)
    bce = nn.BCEWithLogitsLoss()

    best_qwk, best_state, bb_state = -1, None, None
    hist = []
    for ep in range(EPOCHS):
        bb.train(); head.train()
        run_loss, nb = 0.0, 0
        te = time.time()
        for x, Y in train_it.__iter__(augment=True):
            x = x.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                tok = bb(pixel_values=(x - MEAN.cuda()) / STD.cuda(),
                         interpolate_pos_encoding=True).last_hidden_state.float()
            feat = tok[:, 1:].mean(1)
            loss = 0.0
            for ci, c in enumerate(LABEL_COLS):
                y = Y[:, ci].cuda()
                loss = loss + bce(head.heads[c](feat), cumulative_targets(y, N_LEVELS[c]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(bb.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            if sched.last_epoch + 1 < sched.total_steps:
                sched.step()
            run_loss += float(loss); nb += 1
            if nb % 400 == 0:
                print(f"[P2.3] ep{ep} step{nb}/{steps_per_epoch} loss {run_loss/nb:.3f} "
                      f"({time.time()-te:.0f}s)", flush=True)
        m = evaluate(bb, head, val_it, n_va)
        mq = float(np.mean(list(m.values())))
        hist.append({"epoch": ep, "val_mean_qwk": mq,
                     "per_task": {c: round(v, 4) for c, v in m.items()},
                     "train_loss": run_loss / max(nb, 1), "seconds": round(time.time() - te)})
        print(f"[P2.3] ep{ep} val mean QWK {mq:.4f} ({time.time()-te:.0f}s)", flush=True)
        if mq > best_qwk:
            best_qwk = mq
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            bb_state = {k: v.detach().cpu().clone() for k, v in bb.state_dict().items()}
            torch.save({"bb": bb_state, "head": best_state, "val_qwk": mq},
                       PROJ / "results" / "raddino_best_ckpt.pt")

    head.load_state_dict(best_state); bb.load_state_dict(bb_state)
    te_m = evaluate(bb, head, test_it, n_te)
    out = {"config": "RAD-DINO @224 finetune + mean-pool + unconstrained cumulative, 12ep patience3, aug(rot/shift/gamma, no-hflip)",
           "val_best_mean_qwk": best_qwk, "val_history": hist,
           "test": {c: round(v, 4) for c, v in te_m.items()},
           "test_mean_qwk": float(np.mean(list(te_m.values()))),
           "total_seconds": round(time.time() - t0)}
    (PROJ / "results" / "p44_finetune_v2.json").write_text(json.dumps(out, indent=1))
    print(f"[P2.3] : test mean QWK {out['test_mean_qwk']:.4f} "
          f"({out['total_seconds']/60:.0f}min)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        (PROJ / "results" / "p23_finetune_error.json").write_text(
            json.dumps({"error": str(e), "trace": traceback.format_exc()[-2000:]}, indent=1))
        print(f"[P2.3] : {e}", flush=True)
    finally:
        if LOCK.exists():
            LOCK.unlink()
