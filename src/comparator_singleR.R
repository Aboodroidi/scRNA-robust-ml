# SingleR comparator: run on 8K and 3K raw PBMC data using a reference atlas.
#
# Usage:
#   Rscript src/comparator_singleR.R
#
# Outputs:
#   results/comparators/singleR/singleR_8k_predictions.csv
#   results/comparators/singleR/singleR_3k_predictions.csv
#
# Python script comparator_singleR_postprocess.py computes metrics + confusion
# matrices from these prediction files.

suppressPackageStartupMessages({
  library(SingleR)
  library(celldex)
  library(SingleCellExperiment)
  library(scuttle)
  library(Matrix)
})

# ══════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════
RAW_8K_DIR <- "data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38"
RAW_3K_DIR <- "data/raw/pbmc3k/filtered_gene_bc_matrices/hg19"   # will extract below
OUTDIR     <- "results/comparators/singleR"
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

# Extract 3K tarball if needed
if (!dir.exists(RAW_3K_DIR)) {
  tar_3k <- "data/raw/pbmc3k/filtered_gene_bc_matrices.tar.gz"
  if (file.exists(tar_3k)) {
    untar(tar_3k, exdir = "data/raw/pbmc3k/")
  }
}

# Fallback guess for 3K folder structure
if (!dir.exists(RAW_3K_DIR)) {
  candidates <- list.dirs("data/raw/pbmc3k", recursive = TRUE)
  mex_dirs <- candidates[grepl("filtered_gene_bc_matrices/", candidates) &
                          !grepl("^data/raw/pbmc3k/filtered_gene_bc_matrices$", candidates)]
  if (length(mex_dirs) > 0) RAW_3K_DIR <- mex_dirs[1]
}

cat("\n[config] RAW_8K_DIR =", RAW_8K_DIR, "\n")
cat("[config] RAW_3K_DIR =", RAW_3K_DIR, "\n")
cat("[config] OUTDIR     =", OUTDIR, "\n\n")

# ══════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════

# Minimal 10x MEX reader (avoids Seurat dependency)
read_10x_mex <- function(mex_dir) {
  mat_file    <- file.path(mex_dir, "matrix.mtx")
  barcode_file <- file.path(mex_dir, "barcodes.tsv")
  gene_file    <- file.path(mex_dir, "genes.tsv")
  if (!file.exists(gene_file)) gene_file <- file.path(mex_dir, "features.tsv")
  # gz variants
  if (!file.exists(mat_file))     mat_file     <- paste0(mat_file, ".gz")
  if (!file.exists(barcode_file)) barcode_file <- paste0(barcode_file, ".gz")
  if (!file.exists(gene_file))    gene_file    <- paste0(gene_file, ".gz")

  mat <- readMM(mat_file)
  barcodes <- readLines(barcode_file)
  genes_tbl <- read.delim(gene_file, header = FALSE, stringsAsFactors = FALSE)
  gene_symbols <- make.unique(genes_tbl[, 2])
  rownames(mat) <- gene_symbols
  colnames(mat) <- barcodes
  mat
}

run_singleR_on_dataset <- function(counts, ref, out_prefix) {
  cat("  Building SCE…\n")
  sce <- SingleCellExperiment(assays = list(counts = counts))

  cat("  Log-normalising counts…\n")
  sce <- logNormCounts(sce)

  cat("  Running SingleR (this takes a few minutes)…\n")
  t0 <- Sys.time()
  pred <- SingleR(test = sce, ref = ref,
                  labels = ref$label.main,
                  assay.type.test = "logcounts")
  elapsed <- as.numeric(Sys.time() - t0, units = "mins")
  cat(sprintf("  SingleR done in %.1f min\n", elapsed))

  df <- data.frame(
    barcode = colnames(sce),
    predicted_label = pred$labels,
    pruned_label    = pred$pruned.labels,
    score_max       = apply(pred$scores, 1, max)
  )
  out_file <- file.path(OUTDIR, paste0(out_prefix, "_predictions.csv"))
  write.csv(df, out_file, row.names = FALSE)
  cat("  Saved:", out_file, "\n")
  list(elapsed_min = elapsed, n_cells = nrow(df))
}

# ══════════════════════════════════════════════════
# LOAD REFERENCE ATLAS (Monaco Immune — cleanest immune labels)
# ══════════════════════════════════════════════════
cat("Loading Monaco Immune reference atlas (cached after first run)…\n")
ref <- MonacoImmuneData()
cat("  Reference cells:", ncol(ref), "\n")
cat("  Main labels:    ", paste(sort(unique(ref$label.main)), collapse = ", "), "\n\n")

# ══════════════════════════════════════════════════
# 8K
# ══════════════════════════════════════════════════
cat("── PBMC 8K ──\n")
counts_8k <- read_10x_mex(RAW_8K_DIR)
cat("  counts shape:", nrow(counts_8k), "×", ncol(counts_8k), "\n")
info_8k <- run_singleR_on_dataset(counts_8k, ref, "singleR_8k")

# ══════════════════════════════════════════════════
# 3K
# ══════════════════════════════════════════════════
cat("\n── PBMC 3K ──\n")
counts_3k <- read_10x_mex(RAW_3K_DIR)
cat("  counts shape:", nrow(counts_3k), "×", ncol(counts_3k), "\n")
info_3k <- run_singleR_on_dataset(counts_3k, ref, "singleR_3k")

# ══════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════
summary_df <- data.frame(
  dataset     = c("8K", "3K"),
  n_cells     = c(info_8k$n_cells, info_3k$n_cells),
  runtime_min = c(info_8k$elapsed_min, info_3k$elapsed_min),
  reference   = "MonacoImmuneData",
  tool        = "SingleR",
  r_version   = paste(R.version$major, R.version$minor, sep = ".")
)
write.csv(summary_df, file.path(OUTDIR, "singleR_run_info.csv"), row.names = FALSE)

cat("\n✅ SingleR predictions complete.\n")
cat("   Predictions saved to:", OUTDIR, "\n")
cat("   Run Python post-processing next:\n")
cat("     python src/comparator_singleR_postprocess.py\n")
