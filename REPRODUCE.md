# Reproducing every table and figure

Environment: Python 3.10+, a recent CUDA build of `torch`, `transformers`, `scikit-learn`, `pandas`, `pyarrow`, `matplotlib`. Any CUDA GPU with >= 16 GB VRAM works; the linear-head protocol also runs on smaller cards with a reduced batch size. Total GPU budget for all reported experiments: **≈ 18 GPU-hours**.

If `huggingface.co` is unreachable from your network, point `HF_ENDPOINT` to a mirror before the download/model steps.

| Paper table / figure | Script | Result file | Runtime |
|---|---|---|---|
| Data download | `hf download TLAIM/TAIX-Ray --repo-type dataset --include "data/*"` | `taixray/data/` (58.3 GB) | ~20 min |
| Photometric direction/intensity scan (scenario strata; Fig. 3 left) | `code/photometric_scan.py` | `results/photometric_scan.parquet` | ~2 min CPU |
| Feature extraction (224/518, base/large, CLS+patch-mean) | `code/extract_features.py --model {dinov2-base,dinov2-large} --res {224,518} --out features[-518/-large]` | `features*/` | 5–14 min/config |
| Table 3 (frozen matrix; three seeds unless noted) | `code/p25_main_table.py` + RAD-DINO/BiomedCLIP extraction via `code/extract_features.py --model {rad-dino,biomedclip}` | `results/matrix_summary_p4.json` | ~20 min |
| Headline single-best + top-3 ensemble | `code/p49c_ensemble.py` | `results/p49c_ensemble.json` + `results/test_predictions_release.npz` | ~15 min |
| Headline 95% CIs (patient-clustered bootstrap) | `code/p50b_significance.py` | `results/p50b_significance.json` | ~10 min |
| Release npz completion (vitb16/rn50/biomedclip per-image preds) | `code/p49h_release_completion.py` | `results/test_predictions_release.npz` (20 configs) | ~6 min |
| GroupKFold5, budget-matched | `code/p46_cv_matched.py` | `results/p46_cv_matched.json` | ~10 min |
| Table 4 (loss formulations) | `code/train_ordinal.py --loss-matrix` (seeds 0–2) | `results/loss_matrix_p22.json` + `linear_probe_seed{0,1,2}_p22_cs_s{0,1,2}.json` | ~4 min |
| Table 5 + Fig. 2 (calibration + conformal, corrected split protocol) | `code/p24_calibration.py` + `code/p41_conformal_proper.py` | `results/p24_calibration.json` + `results/p41_conformal_proper.json` | ~6 min |
| Fine-tuning, 3-epoch recipe | `code/finetune_p23.py` | `results/p23_finetune.json` | ~105 min |
| Fine-tuning reference recipe (12 epochs, augmented) | `code/finetune_p44_v2.py` | `results/finetune_v2_full_metrics.json` (eval via `code/p34_finetune_metrics.py`) | ~7 h |
| RAD-DINO fine-tune + parity analysis | `code/finetune_p49_raddino.py`; three-metric eval via `code/p49b_raddino_ft_metrics.py` | `results/p44_finetune_v2.json` + `results/raddino_ft_full_metrics.json` | ~7 h + ~12 min |
| Label-space distillation (teacher soft labels → students) | `code/p49g_teacher_labels.py` → `code/p49g_student.py` | `results/teacher_soft_labels.npz` + `results/p49g_student.json` | ~6 min + ~8 min |
| Fig. 3 (scenario stratification + per-grade recall) | `code/p26_stratified.py` | `results/p26_stratified.json` | ~1 min |
| Tables 6–7 (official checkpoint vs. ours, per finding) | `code/p33_official_compare.py` (requires the official `ordinal.ckpt` from HF `TLAIM/TAIX-Ray` placed in `official-ckpt/`, and the [official repo](https://github.com/TruhnLab/TAIX-Ray) cloned next to this project, or `TAIXRAY_OFFICIAL_REPO` set) | `results/p33_official_compare.json` | ~8 min |
| Official baseline decoding ablation (chained vs counting rule) | `code/p33b_decoding_ablation.py` | `results/p33b_decoding_ablation.json` + `results/p33_official_preds.npz` | ~13 min |
| Official vs ours-FT paired patient-clustered CI | `code/p50f_official_paired.py` | `results/p50f_official_vs_ft_paired.json` | ~3 min |
| Figures 1–3 | `code/paper_figures.py` (label helpers from `code/p31_figures.py`) | `figures/Fig{1,2,3}.{pdf,png,tif}` | <1 min |

## Pitfalls (learned the hard way)

1. Loading DINOv2 weights requires `transformers.Dinov2Model` — `ViTModel` silently mis-loads LayerNorm weights and produces constant features.
2. dinov2-base is trained at 518²; 224² inputs need `interpolate_pos_encoding=True` (official linear-probe protocol).
3. Loading someone else's Lightning checkpoint: keys carry module-attribute prefixes (`model.*`) — load them as-is, do not strip.
4. The official checkpoint constructor needs `loss_kwargs={'class_labels_num': [3,4,4,4,4,4,4,4]}`.
5. `OneCycleLR.total_steps` must be computed per-shard (each Parquet shard yields its own tail batch).
6. The released PNGs mix 12-bit and 16-bit intensity scales — always normalize per image (p1/p99 or z-score); never use global absolute-value normalization.
7. CORN heads emit conditional probabilities P(y>k | y>k-1): chain them with a cumulative product before thresholding. Counting thresholded conditionals instead silently understates QWK by a large margin (≈ 0.09 on the official checkpoint).
