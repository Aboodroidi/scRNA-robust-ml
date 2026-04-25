# Reproducing the scRNA-robust-ml Pipeline

Single-command orchestration of the entire dissertation pipeline from raw PBMC data to final figures.

## Usage

From the repo root:

```bash
# Full pipeline (skips training if models already exist)
python reproduce_all.py

# Skip individual stages
python reproduce_all.py --skip-training --skip-preprocess

# Only regenerate figures
python reproduce_all.py --figures-only

# See what would run without executing
python reproduce_all.py --dry-run

# Force full retrain (~4 hours)
python reproduce_all.py --full-retrain
```

## Expected Runtime

Tested on MacBook (Apple Silicon, CPU-only, 16 GB RAM):

| Mode | Time |
|------|------|
| Full retrain (`--full-retrain`) | ~4–5 hours (XGB_HVG alone is ~3h) |
| Default (models already trained) | ~25 min (eval + calibration + ablation + figures) |
| Figures only (`--figures-only`) | ~3 min |
| Robustness stage (`--skip-robustness` to skip) | ~20 min |

## Pipeline Stages

| # | Stage | Script(s) Called |
|---|-------|------------------|
| 1 | Preprocessing | `src/preprocess.py`, `src/annotate.py`, `src/preprocess_pbmc8k.py`, `src/preprocess_pbmc3k.py` |
| 2 | HVG training | `src/train_all_full_models.py` |
| 3 | PCA training | `src/train_all_pca_models.py` |
| 4 | Cross-donor eval | `src/evaluate_external_pbmc8k.py`, `src/evaluate_external_pbmc3k.py` |
| 5 | Calibration | `src/plot_temp_scaling_all.py`, `src/plot_reliability_all_models.py` |
| 6 | Ablation | `src/ablation_mask_pca.py` |
| 7 | Robustness | `src/robustness_cross_donor.py` |
| 8 | Figures | ~20 individual plotting scripts under `src/plot_*.py` |

## Output Files

| Location | Contents |
|----------|----------|
| `data/processed/*.h5ad` | Preprocessed AnnData for 68K, 8K, 3K |
| `results/full_train_all_hvg_coarse/` | Trained LR, XGB, SANN (HVG) |
| `results/full_train_all_pca_coarse/` | Trained LR, XGB, SANN (PCA) |
| `results/external_validation/` | 8K cross-donor results |
| `results/external_validation_3k/` | 3K cross-donor results |
| `results/ablation_mask/` | Mask ablation results |
| `results/figures/` | All PNG figures at 300 DPI |

## Troubleshooting

**Missing raw data.** The 68K/8K/3K raw files must be present under `data/raw/`. If preprocessing fails with `FileNotFoundError`, download from 10x Genomics:
- 68K PBMCs: https://www.10xgenomics.com/resources/datasets
- 8K PBMCs (Donor B), 3K PBMCs (Donor C): from the same repository

**Intel MKL warnings.** These are harmless on Apple Silicon; the script sets `KMP_DUPLICATE_LIB_OK=TRUE` automatically.

**"Models already exist" message.** By default the script skips training if `lr_model.pkl` is found in the output directory. Use `--full-retrain` to force retraining.

**XGB_HVG takes 3+ hours.** This is expected — XGBoost on 2000 HVG features is slow. Use `--skip-training` after the first successful run.

**Script halts mid-stage.** Every stage is independent; fix the underlying error and re-run with appropriate `--skip-*` flags to jump past completed stages.

## Hardware

Developed and tested on:
- MacBook Pro, Apple Silicon
- Python 3.9 (Anaconda environment)
- 16 GB unified memory, CPU-only (no CUDA)

See `requirements.txt` for exact package versions.
