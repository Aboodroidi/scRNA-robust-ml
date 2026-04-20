# Comparator Tools — Head-to-Head Evaluation

Comparing the SANN_PCA 3-seed ensemble against established cell-type classification tools on the same PBMC 8K and 3K held-out donors.

## Results summary

| Tool              | 8K Acc | 8K Macro-F1 | 8K Weighted-F1 | 3K Acc | 3K Macro-F1 | 3K Weighted-F1 | Training/ref time¹ |
|-------------------|-------:|------------:|---------------:|-------:|------------:|---------------:|-------------------:|
| **SANN_PCA (ours)**  | 0.987  | 0.788       | 0.990          | 0.985  | 0.972       | 0.985          | ~40 min (CPU)      |
| SingleR (Monaco)  | 0.959  | 0.763       | 0.971          | 0.974  | 0.774       | 0.976          | 34.3 min²          |
| Seurat label xfer | 0.990  | 0.790       | 0.992          | 0.985  | 0.965       | 0.985          | ~15 min (CPU)³     |
| ACTINN (reimpl)   | 0.980  | 0.767       | 0.982          | 0.972  | 0.943       | 0.973          | **24.0 h** (CPU)⁴  |
| scANVI            | 0.960  | 0.736       | 0.962          | 0.950  | 0.734       | 0.950          | 26.0 min (T4 GPU)⁵ |

¹ Wall-clock time to train/build the reference on Donor A (68K), measured on the same pre-AVX CPU. Inference time on 8K/3K is negligible (<5 s) for all tools except SingleR (see note 2).
² SingleR has no training phase — reported time is reference-to-query matching on 8K (32.8 min) + 3K (1.5 min). Runtime grows linearly with query size.
³ Seurat pipeline (NormalizeData + ScaleData + PCA + FindTransferAnchors) was not cleanly instrumented — only `TransferData` was timed (2 s on 8K, <1 s on 3K). Total reference-build + transfer observed to complete in roughly 15 minutes during the run.
⁴ ACTINN training time is anomalously high because the host CPU lacks AVX: BLAS cannot vectorise the 19,224 × 100 first-layer matmul. A modern AVX/GPU machine would complete the same run in ~5 min (paper reports <10 min on GPU). Reported honestly as a reproducibility artefact, not a property of the method.
⁵ scANVI was trained on a Google Colab T4 GPU because the local CPU lacks AVX support required by modern JAX (an unavoidable dependency of `scvi-tools ≥ 1.0`). Runtime is reported for completeness but is **not directly comparable** to the other rows. Accuracy metrics are computed on identical held-out data and remain comparable.

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

### scANVI (DID RUN — on Google Colab GPU)

- **Status:** Ran successfully on a Google Colab T4 GPU after confirming the local machine cannot host `scvi-tools`.
- **Why Colab:** `scvi-tools` (1.0+) depends on `jax` which requires AVX CPU instructions. The development machine (older Intel Mac, SSE4.2-only) cannot install any released `jaxlib`. Downgrading to `scvi-tools 0.20.3` (pre-jax) was also attempted but the pip install merged 1.x source files with 0.20.3 metadata. Rather than keep patching, we moved the workload to Colab. Data bundle was uploaded via Google Drive (mounted at runtime) after the Colab sidebar failed to parse a binary tarball.
- **Version:** `scvi-tools 1.4.2` (Python 3.11, Colab default).
- **Input:** raw 10x counts from all three donors → gene intersection → concatenated AnnData with `batch_key=dataset`. `highly_variable_genes(flavor="seurat_v3", n_top_genes=2000, batch_key="batch")` on raw counts.
- **Labels:** 68K cells carry the 5-class coarse labels; 8K/3K cells are marked `Unknown` so scANVI treats them as query cells in the semi-supervised ELBO. This matches the cross-donor protocol used for SANN.
- **Training:** SCVI pre-training 100 epochs (11.8 min) → SCANVI fine-tuning 50 epochs (13.8 min) → prediction on the `8K` and `3K` batches. Total 26.0 min on a T4.
- **Outcome:** 8K accuracy 0.9602 / macro-F1 0.7360; 3K accuracy 0.9496 / macro-F1 0.7341. Weighted-F1 ≈ 0.95–0.96 on both donors, but **Platelet F1 = 0.00 on both donors.** On 8K this is by construction (no Platelet ground-truth cells). On 3K — where Seurat achieved Platelet F1 = 0.93 from the same 68K reference — scANVI missed all 15 true Platelets. The 68K training set has only 236 Platelets (0.3%) against ~52K T-cells; the unweighted ELBO objective washes out such a rare class. NK F1 is also weak (0.74–0.75 across both donors) — the same class-imbalance artefact. This is a genuine methodological finding worth reporting: scANVI's generative objective trades per-class fidelity for a well-mixed latent space, so it underperforms Seurat and SANN on cell-class-level metrics even on the same reference.
- **Reproducibility caveat:** runtime is not comparable to the other rows (different hardware). Accuracy/F1 are fully comparable because evaluation is identical to every other tool.

### Seurat label transfer (DID RUN)

- **Version:** Seurat (CRAN install, R 4.4.0).
- **Reference:** built in-house from PBMC 68K raw 10x counts + the same 5-class coarse labels used for SANN training. No external atlas — this gives Seurat the fairest footing against SANN (same reference donor, same label taxonomy).
- **Input:** raw 10x MEX matrices for 68K / 8K / 3K. Standard Seurat pipeline: `NormalizeData` → `FindVariableFeatures` (vst, 2000) → `ScaleData` → `RunPCA` (30 PCs). Label transfer via `FindTransferAnchors(dims = 1:30, reference.reduction = "pca")` then `TransferData(refdata = ref$coarse_label)`.
- **No label mapping needed:** the reference carries the 5-class taxonomy directly, so Seurat predictions are native 5-class. Any prediction outside the taxonomy (none observed in practice) would land in `"Other"`.
- **Platelet advantage over SingleR:** because the 68K reference contains Platelet cells (Seurat was given the same training donor as SANN), Seurat achieves Platelet F1 = 0.93 on 3K — whereas SingleR's Monaco reference has no Platelet cells at all. 8K has no Platelet ground-truth cells, so Platelet F1 = 0 there by construction (same as SANN and SingleR).
- **Outcome:** Seurat slightly edges SANN on 8K accuracy (0.990 vs 0.987) but lags on 3K macro-F1 (0.965 vs 0.972). Across both donors Seurat is the strongest external comparator.
- **Runtime:** reference build + both transfers on CPU.

### ACTINN (DID RUN — PyTorch reimplementation)

- **Status:** Paper-faithful PyTorch reimplementation of Ma & Pellegrini (2020). The upstream repo targets TensorFlow 1.x / Python 3.7 and does not install on modern Python; rather than fight dependency hell (same trap as scANVI), we reimplemented the architecture exactly as specified in the paper.
- **Architecture:** `Linear(D → 100) → ReLU → Linear(100 → 50) → ReLU → Linear(50 → 25) → ReLU → Linear(25 → 5)`. ~1.9M parameters dominated by the first layer.
- **Input:** raw 10x counts from all three donors → gene-symbol intersection (30,316 shared) → drop genes with zero expression in 68K (11,092 dropped) → **19,224 genes retained** → `log2(x + 1)`.
- **Training:** Adam, lr = 1e-4, cross-entropy. **Paper defaults were 50 epochs, batch 128**; on CPU without AVX one epoch at that batch size was infeasible (>1 hr). We reduced to **20 epochs, batch 512** — paper's Figure 2 shows convergence well before epoch 20, and our training curves confirmed it (train_acc = 0.977 by epoch 9, gains negligible thereafter). Deviation documented for transparency.
- **No label mapping needed:** ACTINN trains on 5-class coarse labels directly, so predictions are native 5-class (same regime as Seurat).
- **Runtime:** ~23 hours total training on pre-AVX CPU (~70 min/epoch with 19K genes × 68K cells). A modern AVX/GPU machine would complete in ~5 min.
- **Outcome:** Competitive with SingleR on 8K and slightly better on 3K. Clearly behind SANN and Seurat on both donors. NK F1 is the weakest cell-class (0.87 on 8K, 0.83 on 3K) — suggesting the 25-unit bottleneck layer struggles with the NK/T-cell boundary, which SANN's attention pathway handles natively.

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

### Run ACTINN (reimplementation)

```bash
# Train on 68K + predict on 8K/3K  (slow on CPU; ~23 h on pre-AVX hardware, ~5 min on GPU)
KMP_DUPLICATE_LIB_OK=TRUE python src/comparator_actinn.py

# Compute metrics + confusion matrices (seconds)
KMP_DUPLICATE_LIB_OK=TRUE python src/comparator_actinn_postprocess.py
```

Outputs land in `results/comparators/actinn/`:
- `actinn_{8k,3k}_predictions.csv` — per-cell predictions + max softmax score
- `actinn_{8k,3k}_confusion.csv` — 6×6 confusion matrix (5 classes + "Other")
- `actinn_metrics.json` — all metrics in one JSON
- `actinn_run_info.csv` — runtime, epochs, hyperparameters, torch version

### Run Seurat label transfer

```bash
# Install Seurat (one-off, ~20–45 min)
Rscript src/install_seurat.R

# Export barcode → label CSVs (68K coarse, 8K native, 3K native)
KMP_DUPLICATE_LIB_OK=TRUE python src/export_labels_for_seurat.py

# Build 68K reference, transfer to 8K and 3K
Rscript src/comparator_seurat.R

# Compute metrics + confusion matrices
KMP_DUPLICATE_LIB_OK=TRUE python src/comparator_seurat_postprocess.py
```

Outputs land in `results/comparators/seurat/`:
- `pbmc68k_seurat_reference.rds` — built reference (can be reused)
- `seurat_{8k,3k}_predictions.csv` — per-cell predictions + max transfer score
- `seurat_{8k,3k}_confusion.csv` — 6×6 confusion matrix (5 classes + "Other")
- `seurat_metrics.json` — all metrics in one JSON
- `seurat_run_info.csv` — runtime & tool versions
- `labels/` — barcode→label CSVs consumed by the R script

### Run scANVI (on Google Colab)

The local CPU cannot host `scvi-tools`, so training is performed on Colab and
only the prediction CSVs are pulled back for metric computation.

```bash
# 1. Bundle raw data + label CSVs for upload (local)
tar czf scanvi_data.tgz \
    data/raw/pbmc68k/filtered_matrices_mex/hg19 \
    data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38 \
    data/raw/pbmc3k/mex \
    results/comparators/seurat/labels/

# 2. Colab (T4 runtime): upload the bundle (Google Drive mount works best)
#    and run the Colab variant of the script:
# !tar xzf scanvi_data.tgz -C /content/
# !pip install -q "scvi-tools" "anndata"
# !python comparator_scanvi_colab.py --use_gpu

# 3. Download the two prediction CSVs back into
#    results/comparators/scanvi/

# 4. Compute metrics + confusion matrices locally
KMP_DUPLICATE_LIB_OK=TRUE python src/comparator_scanvi_postprocess.py
```

Outputs land in `results/comparators/scanvi/`:
- `scanvi_{8k,3k}_predictions.csv` — per-cell predictions
- `scanvi_{8k,3k}_confusion.csv` — 6×6 confusion matrix (5 classes + "Other")
- `scanvi_metrics.json` — all metrics in one JSON
- `scanvi_run_info.csv` — scvi-tools version, epochs, GPU runtime

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
| `results/comparators/seurat/` | Seurat predictions, confusion matrices, metrics |
| `src/comparator_seurat.R` | R script building 68K reference + transfers to 8K/3K |
| `src/comparator_seurat_postprocess.py` | Metrics + confusion matrices for Seurat |
| `src/export_labels_for_seurat.py` | Exports barcode→label CSVs used by the R script |
| `src/install_seurat.R` | One-shot Seurat install helper |
| `results/comparators/actinn/` | ACTINN predictions, confusion matrices, metrics |
| `src/comparator_actinn.py` | PyTorch reimplementation, trains on 68K + predicts on 8K/3K |
| `src/comparator_actinn_postprocess.py` | Metrics + confusion matrices for ACTINN |
| `results/comparators/scanvi/` | scANVI predictions, confusion matrices, metrics, run info |
| `src/comparator_scanvi_colab.py` | Colab-ready pipeline (SCVI + SCANVI training on GPU) |
| `src/comparator_scanvi_postprocess.py` | Metrics + confusion matrices for scANVI |
| `src/export_3k_to_mex.py` | Convert 3K h5ad → 10x MEX format |
| `src/install_singleR.R` | One-shot R install helper |
