# SingleR 3K-only rerun — uses the exported MEX files from pbmc3k_raw.h5ad.
# Run after `python src/export_3k_to_mex.py`.

suppressPackageStartupMessages({
  library(SingleR)
  library(celldex)
  library(SingleCellExperiment)
  library(scuttle)
  library(Matrix)
})

RAW_3K_DIR <- "data/raw/pbmc3k/mex"
OUTDIR     <- "results/comparators/singleR"
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

read_10x_mex <- function(mex_dir) {
  mat_file     <- file.path(mex_dir, "matrix.mtx.gz")
  barcode_file <- file.path(mex_dir, "barcodes.tsv.gz")
  gene_file    <- file.path(mex_dir, "features.tsv.gz")
  mat <- readMM(mat_file)
  barcodes <- readLines(barcode_file)
  genes_tbl <- read.delim(gene_file, header = FALSE, stringsAsFactors = FALSE)
  gene_symbols <- make.unique(genes_tbl[, 2])
  rownames(mat) <- gene_symbols
  colnames(mat) <- barcodes
  mat
}

cat("Loading Monaco Immune reference (cached)…\n")
ref <- MonacoImmuneData()

cat("\n── PBMC 3K ──\n")
counts_3k <- read_10x_mex(RAW_3K_DIR)
cat("  counts shape:", nrow(counts_3k), "×", ncol(counts_3k), "\n")

sce <- SingleCellExperiment(assays = list(counts = counts_3k))
sce <- logNormCounts(sce)

cat("  Running SingleR…\n")
t0 <- Sys.time()
pred <- SingleR(test = sce, ref = ref,
                labels = ref$label.main,
                assay.type.test = "logcounts")
elapsed <- as.numeric(Sys.time() - t0, units = "mins")
cat(sprintf("  Done in %.1f min\n", elapsed))

df <- data.frame(
  barcode         = colnames(sce),
  predicted_label = pred$labels,
  pruned_label    = pred$pruned.labels,
  score_max       = apply(pred$scores, 1, max)
)
out_file <- file.path(OUTDIR, "singleR_3k_predictions.csv")
write.csv(df, out_file, row.names = FALSE)
cat("  Saved:", out_file, "\n")

# Update run info
run_info_path <- file.path(OUTDIR, "singleR_run_info.csv")
if (file.exists(run_info_path)) {
  ri <- read.csv(run_info_path, stringsAsFactors = FALSE)
} else {
  ri <- data.frame()
}
new_row <- data.frame(
  dataset     = "3K",
  n_cells     = ncol(sce),
  runtime_min = elapsed,
  reference   = "MonacoImmuneData",
  tool        = "SingleR",
  r_version   = paste(R.version$major, R.version$minor, sep = ".")
)
ri <- rbind(ri[ri$dataset != "3K", , drop = FALSE], new_row)
write.csv(ri, run_info_path, row.names = FALSE)

cat("\n✅ 3K SingleR predictions complete.\n")
