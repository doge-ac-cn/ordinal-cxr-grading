# Ordinal CXR Grading

Benchmark and systematic study of ordinal output parameterization on bedside chest radiographs, built on [TAIX-Ray](https://doi.org/10.1038/s41597-026-07271-7): 215,381 portable AP radiographs from 47,724 ICU patients, 8 findings with 4–5-grade severity labels from prospective structured reporting.

## Headline results

| Model | Mean QWK | Linear κ | Exact acc. | Training cost |
|---|---|---|---|---|
| Official TAIX-Ray baseline (their ckpt, our eval) | **0.6115** | **0.5117** | **0.6310** | ≈11 GPU-h |
| Frozen single-best (20-config per-task validation selection) | 0.5463 [0.5434–0.5493] | 0.4351 | 0.5328 | ≈25 GPU-min |
| **Frozen top-3 ensemble** | **0.5660** [0.5629–0.5691] | 0.4449 | 0.5320 | ≈25 GPU-min |
| Fine-tuned, 3-epoch recipe (DINOv2-B, 224², cumulative head) | 0.5973 | 0.4982 | 0.6238 | ≈1.7 GPU-h |
| Fine-tuned reference recipe (12 epochs, augmented) | 0.6044 | 0.5053 | 0.6274 | ≈7 GPU-h |

Key findings:

1. **Output parameterization is the first-order design choice**: cumulative heads with rank-consistent inference beat plain CE by +0.033 QWK; soft-label smoothing recovers only +0.007. CORAL and CORN land between. The advantage carries into calibration (2× lower ECE); conformal validity holds for every parameterization.
2. **Inference decoding is part of the method**: a CORN head emits conditional probabilities P(y>k | y>k-1) and must be decoded by chaining them (cumulative product). Threshold-counting the conditionals instead costs ≈0.09 QWK on the official checkpoint and flips its standing relative to our best fine-tuned models. All CORN numbers here use the chaining rule.
3. **Radiology-adapted pretraining and native resolution dominate frozen performance**: RAD-DINO reaches 0.5371 (+0.103 over the best generic encoder, wins all 8 validation selections); native 518² DINOv2 beats 224² (+0.02); ViT-L underperforms ViT-B at 224². Neither advantage survives fine-tuning (RAD-DINO FT 0.6022 ≈ generic FT 0.6044).
4. **Extreme severity grades are a resolution ceiling**: top-grade recall is near zero under frozen features—quantified, not hidden.
5. **Label-space distillation transfers only a sliver**: cum-space KD students gain +0.004–0.010 over hard-label probes, far below the fine-tuned layer.

## Repository structure

```
code/        all experiment scripts (see results/ directory and REPRODUCE.md for the table→script→result map)
results/     every number in the paper, as JSON (108 files)
             + per-image test predictions for all 20 frozen configurations (test_predictions_release.npz)
             + per-image test predictions for the RAD-DINO fine-tune (p49_raddino_test_preds.npz)
             + teacher soft labels for distillation (teacher_soft_labels.npz)
             + photometric scan table (photometric_scan.parquet)
figures/     paper figures Fig1–3 (PDF + PNG 600 dpi + TIF), regenerable via code/paper_figures.py
LICENSE      MIT (code); dataset is CC-BY-4.0 (see below)
```

## Setup

```bash
conda create -n taixray python=3.12 -y && conda activate taixray
pip install torch==2.7.1 torchvision pandas pyarrow scikit-learn matplotlib transformers
# optional: export HF_ENDPOINT=<a mirror> if huggingface.co is unreachable from your network
```

Any CUDA GPU with ≥16 GB VRAM works (smaller cards are fine for the linear-head protocol at reduced batch size). Total GPU budget for every experiment in the paper: ≈ 18 GPU-hours.

## Data

```bash
hf download TLAIM/TAIX-Ray --repo-type dataset --include "data/*" --local-dir taixray
```

58.3 GB (Parquet, patient-disjoint official splits: 137,593 / 34,860 / 42,928). The 1 TB `original` config is not needed.

## Quickstart

```bash
# 1. extract frozen DINOv2 features (CLS + patch-mean), 3 GPUs, ~5 min
for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i python3 code/extract_features.py \
  --shard-idx $i --num-shard-groups 3 --model dinov2-base --res 518 --out features-518 & done; wait

# 2. train the cumulative head (headline protocol)
python3 code/train_ordinal.py --epochs 100 --seed 0 --features-dir features-518 --pool cls

# 3. reproduce every paper table
#    see results/ directory and REPRODUCE.md for the full table → script → result mapping
```

## Notable pitfalls (learned the hard way — see results/ directory and REPRODUCE.md)

- Loading DINOv2 weights requires `transformers.Dinov2Model`; `ViTModel` silently mis-loads LayerNorms and produces constant features.
- The released PNGs mix 12-bit and 16-bit intensity scales — per-image percentile (or z-score) normalization is mandatory.
- PyTorch Lightning checkpoints key modules with attribute prefixes (`model.*`); stripping them yields a randomly-initialized network that looks like "no signal".
- CORN heads emit conditional probabilities P(y>k | y>k-1): chain them with a cumulative product before thresholding. Counting thresholded conditionals instead silently understates QWK by a large margin.

## Citation

```bibtex
@article{taixray2026,
  title={A comprehensive bedside chest radiography dataset with structured,
         itemized and graded radiologic reports},
  author={Truhn, Daniel and others},
  journal={Scientific Data},
  year={2026},
  doi={10.1038/s41597-026-07271-7}
}
```

Benchmark paper citation: *to be added upon publication*.

## License

Code: MIT. The TAIX-Ray dataset is CC-BY-4.0. Evaluation of the official baseline checkpoint follows the terms accompanying that release.
