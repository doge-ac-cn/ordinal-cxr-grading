"""TAIX-Ray DINOv2

:
  CUDA_VISIBLE_DEVICES=0 python extract_features.py --shard-idx 0 --num-shard-groups 3 --split all
  CUDA_VISIBLE_DEVICES=1 python extract_features.py --shard-idx 1 --num-shard-groups 3 --split all
  CUDA_VISIBLE_DEVICES=2 python extract_features.py --shard-idx 2 --num-shard-groups 3 --split all

:
-  (p1/p99)12/16
-  resize  224x224 center crop
- DINOv2-base CLS token (768d, fp16)
"""
import argparse, io, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1] / "taixray" / "data"
PROJ = Path(__file__).resolve().parents[1]
LABEL_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right",
              "PleuralEffusion_Left", "PulmonaryOpacities_Right", "PulmonaryOpacities_Left",
              "Atelectasis_Right", "Atelectasis_Left"]
META_COLS = ["UID", "PatientID", "Fold", "Split"] + LABEL_COLS
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def decode_normalize(img_bytes: bytes, invert: bool = False, res: int = 224) -> Image.Image:
    """16PNG -> () -> p1/p99uint8 -> 3"""
    arr = np.asarray(Image.open(io.BytesIO(img_bytes))).astype(np.float32)
    if invert:
        arr = arr.max() - arr
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    if p99 <= p1:
        p1, p99 = arr.min(), max(arr.max(), arr.min() + 1)
    arr = np.clip((arr - p1) / (p99 - p1), 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB").resize((res, res), Image.BILINEAR)


class ShardDS(Dataset):
    def __init__(self, path: Path, invert_uids: set, res: int = 224):
        self.res = res
        t = pq.read_table(path, columns=["Image"] + META_COLS)
        self.imgs = t.column("Image").to_pylist()
        self.invert = [u in invert_uids for u in t.column("UID").to_pylist()]
        self.meta = t.select(META_COLS).to_pandas()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = decode_normalize(self.imgs[i]["bytes"], self.invert[i], self.res)
        return i, torch.from_numpy(np.asarray(x).transpose(2, 0, 1).astype(np.float32) / 255.0),


def norm_batch(x: torch.Tensor, mean=None, std=None) -> torch.Tensor:
    if mean is None:
        mean, std = MEAN.to(x.device), STD.to(x.device)
    return (x - mean) / std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--num-shard-groups", type=int, default=3)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="N()")
    ap.add_argument("--model", default="dinov2-base",
                    choices=["dinov2-base", "dinov2-large", "resnet50", "vitb16-imagenet", "rad-dino", "biomedclip"])
    ap.add_argument("--out", default="features", help="()")
    ap.add_argument("--res", type=int, default=224, help="(224/518)")
    args = ap.parse_args()

    OUT = PROJ / args.out
    OUT.mkdir(exist_ok=True)
    device = "cuda"
    kind = args.model
    from transformers import Dinov2Model
    if kind == "biomedclip":
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        model = model.to(device).eval()
        dim = 512
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    elif kind == "rad-dino":
        model = Dinov2Model.from_pretrained("microsoft/rad-dino").to(device).eval()
        dim = 768
    elif kind.startswith("dinov2"):
        model = Dinov2Model.from_pretrained(f"facebook/{args.model}").to(device).eval()
        dim = 768 if kind == "dinov2-base" else 1024
    elif kind == "resnet50":
        import torch.nn as nn
        import torchvision.models as tvm
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()
        model = m.to(device).eval()
        dim = 2048
    elif kind == "vitb16-imagenet":
        from transformers import ViTModel
        model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device).eval()
        dim = 768
    for p in model.parameters():
        p.requires_grad_(False)

    scan = pd.read_parquet(Path(__file__).resolve().parents[1] / "results" / "photometric_scan.parquet")
    invert_uids = set(scan.loc[(scan.spine_ratio < 0.9) & (scan.frac_high > 0.15), "UID"])
    print(f"[proc{args.shard_idx}] :  {len(invert_uids)} ", flush=True)

    shards = []
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for s in splits:
        shards += sorted(ROOT.glob(f"{s}-*.parquet"))
    mine = [p for n, p in enumerate(shards) if n % args.num_shard_groups == args.shard_idx]
    print(f"[proc{args.shard_idx}]  {len(mine)}/{len(shards)} ", flush=True)

    t_start = time.time()
    n_done = 0
    for path in mine:
        out_npz = OUT / f"{path.stem}.npz"
        if out_npz.exists():
            continue
        ds = ShardDS(path, invert_uids, args.res)
        dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers)
        feats = np.zeros((len(ds), dim), dtype=np.float16)
        feats_mean = np.zeros((len(ds), dim), dtype=np.float16)
        with torch.autocast("cuda", dtype=torch.float16):
            for idx, x in dl:
                with torch.no_grad():
                    if kind == "biomedclip":
                        xin = norm_batch(x, clip_mean, clip_std).to(device, non_blocking=True)
                    else:
                        xin = norm_batch(x).to(device, non_blocking=True)
                    if kind == "biomedclip":
                        out = model.encode_image(xin)
                    elif kind == "resnet50":
                        out = model(xin)
                    else:
                        out = model(pixel_values=xin,
                                    interpolate_pos_encoding=True)
                if kind == "resnet50":
                    feats[idx.numpy()] = out.float().cpu().numpy().astype(np.float16)
                elif kind == "biomedclip":
                    feats[idx.numpy()] = out.float().cpu().numpy().astype(np.float16)
                else:
                    tok = out.last_hidden_state.float()
                    feats[idx.numpy()] = tok[:, 0].cpu().numpy().astype(np.float16)
                    feats_mean[idx.numpy()] = tok[:, 1:].mean(1).cpu().numpy().astype(np.float16)
        keep = slice(0, args.limit) if args.limit else slice(None)
        if kind in ("resnet50", "biomedclip"):
            np.savez_compressed(out_npz, feats=feats[keep])
        else:
            np.savez_compressed(out_npz, feats=feats[keep], feats_mean=feats_mean[keep])
        ds.meta.iloc[keep].to_parquet(OUT / f"{path.stem}.meta.parquet")
        n_done += len(ds)
        el = time.time() - t_start
        print(f"[proc{args.shard_idx}] {path.stem}: {len(ds)}  |  {n_done}  | "
              f"{n_done/el:.0f} img/s |  {el/60:.1f}min", flush=True)
    print(f"[proc{args.shard_idx}] ", flush=True)


if __name__ == "__main__":
    main()
