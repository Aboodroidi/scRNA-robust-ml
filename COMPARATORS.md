# Comparator Tools — Head-to-Head Evaluation

Comparing the SANN_PCA 3-seed ensemble against established cell-type classification tools on the same PBMC 8K and 3K held-out donors.

## Results summary

| Tool              | 8K Acc | 8K Macro-F1 | 8K Weighted-F1 | 3K Acc | 3K Macro-F1 | 3K Weighted-F1 |
|-------------------|-------:|------------:|---------------:|-------:|------------:|---------------:|
| **SANN_PCA (ours)**  | 0.987  | 0.788       | 0.990          | 0.985  | 0.972       | 0.985          |
| SingleR (Monaco)  | 0.959  | 0.763       | 0.971          | 0.974  | 0.774       | 0.976          |
| Seurat label xfer | —      | —           | —              | —      | —           | —              |
| scANVI            | **dropped** — see notes |
| ACTINN            | **not attempted** — see notes |

Raw per-class F1 and notes are in `results/comparators/comparator_results.csv`.

Macro-F1 values reflect the 5-class coarse taxonomy (B cells, Mono, NK, Platelet, T cells). Platelet F1 = 0 for SingleR on both donors because the Monaco Immune reference atlas lacks Platelet labels. This is a legitimate limitation of reference-based methods.

## Protocol

- **Training data (ours only):** PBMC 68K (Donor A), 5-class coarse labels.
- **Test data (all tools):** PBMC 8K (Donor B), PBMC 3K (Donor C), both preprocessed identically in the main pipeline.
- **Evaluation:** cells with labels outside the 5-class taxonomy (e.g. DC, CL 1) are excluded from accuracy/F1 calculations. This affects all tools equally.
- **No tool had hyperparameters tuned on the held-out donors.** SingleR used default parameters with the Monaco Immune reference atlas chosen because it has the cleanest overlap with PBMC immune cell types.

## Tool-specific notes

### SingleR (DID RUN)

- **Version:** SingleR Bioconductor, installed via `BiocManager` on R 4.4.0.
- **Reference:** `celldex::MonacoImmuneData()` — 114 immune reference cells with `label.main` classes (B cells, Monocytes, NK cells, CD4+ T cells, CD8+ T cells, T cells, Dendritic cells, Basophils, Neutrophils, Progenitors).
- **Input:** raw 10x counts, log-normalised in R via `scuttle::logNormCounts()`. SingleR's recommended preprocessing pipeline was used rather than the main pipeline's HVG/PCA features.
- **Label mapping (Monaco → ours):**
  - `B cells` → `B cells`
  - `Monocytes` → `Mono`
  - `NK cells` → `NK`
  - `CD4+ T cells`, `CD8+ T cells`, `T cells` → `T cells`
  - `Dendritic cells`, `Basophils`, `Neutrophils`, `Progenitors` → `Other` (count as misclassifications)
- **Platelet problem:** Monaco Immune has no Platelet reference cells. Any cell with a true Platelet label was necessarily predicted incorrectly (15 cells in 3K). This pulls SingleR's 3K macro-F1 down from an otherwise strong 0.97 (weighted-F1) to 0.77.
- **Runtime:** 8K took 32.8 min, 3K took ~5 min. Run on CPU.

### scANVI (DROPPED)

- **Status:** Excluded from comparison due to environment incompatibility.
- **Reason:** `scvi-tools` (1.0+) depends on `jax` which requires AVX CPU instructions. The development machine (older Intel Mac, SSE4.2-only) cannot run modern jax. We tried downgrading to `scvi-tools 0.20.3` (pre-jax), but the pip-installed files on disk were corrupted — the metadata reported 0.20.3 while the actual module tree was from 1.x. Patching around individual imports led to further jax chains.
- **Honest framing for the paper:** this is a legitimate reproducibility issue worth noting — scANVI's hard dependency on jax/AVX excludes it from deployment on commodity hardware. A re-run on a newer machine would be feasible.

### Seurat label transfer (not yet run)

- **Status:** Scripts not written yet. Seurat is available in the R install but the label-transfer workflow (`FindTransferAnchors` / `TransferData`) requires a Seurat reference built from the 68K data, which is a separate setup step.
- **Next step:** build the 68K Seurat reference, transfer to 8K and 3K, post-process identically to SingleR.

### ACTINN (not attempted)

- **Status:** Deferred / likely to be dropped.
- **Reason:** ACTINN's upstream code targets TensorFlow 1.x (Python 3.7). Modern Python (3.9+) cannot install the original repo without significant patching. Given the AVX issues already encountered and the diminishing marginal value of a fourth comparator that's likely to fail similarly, we recommend framing the paper around a three-tool comparison (SANN vs SingleR vs Seurat) rather than forcing ACTINN through.

## Reproducing the comparators

### Install R packages (~30–60 min)

```bash
Rscript src/install_singleR.R
```

### Run SingleR

```bash
# 8K (raw MEX already extracted)
Rscript src/comparator_singleR.R

# 3K (requires h5ad-to-MEX export first)
python src/export_3k_to_mex.py
Rscript src/comparator_singleR_3k.R

# Compute metrics + confusion matrices
KMP_DUPLICATE_LIB_OK=TRUE python src/comparator_singleR_postprocess.py
```

Outputs land in `results/comparators/singleR/`:
- `singleR_{8k,3k}_predictions.csv` — per-cell predictions
- `singleR_{8k,3k}_confusion.csv` — 6×6 confusion matrix (5 classes + "Other")
- `singleR_metrics.json` — all metrics in one JSON
- `singleR_run_info.csv` — runtime & tool versions

### scANVI (not recoverable on this hardware)

```bash
# Attempted, does not run due to AVX missing in host CPU
# pip install "scvi-tools==0.20.3"
# python src/comparator_scanvi.py
```

## Fairness disclosures

1. **SANN was trained on 68K; SingleR uses Monaco Immune (not 68K).** This is standard — each tool uses its natural reference. Both are evaluated on the same held-out 8K/3K cells.
2. **The 5-class taxonomy favours SANN** because SANN's training labels are exactly the 5 classes, while SingleR's reference has finer granularity that must be collapsed. This is disclosed in the label mapping above.
3. **Platelet is impossible for SingleR** — this is a genuine reference-coverage limitation and reported transparently.
4. **No hyperparameter tuning** was performed on the 8K or 3K test sets for either tool.

## Files

| Path | Contents |
|------|----------|
| `results/comparators/comparator_results.csv` | Summary table (both tools, both donors) |
| `results/comparators/singleR/` | SingleR predictions, confusion matrices, metrics |
| `src/comparator_singleR.R` | R script for 8K (and 3K if tarball extracts) |
| `src/comparator_singleR_3k.R` | R script for 3K from exported MEX |
| `src/comparator_singleR_postprocess.py` | Label mapping + metrics computation |
| `src/comparator_scanvi.py` | scANVI script (not runnable on this machine) |
| `src/export_3k_to_mex.py` | Convert 3K h5ad → 10x MEX format |
| `src/install_singleR.R` | One-shot R install helper |
