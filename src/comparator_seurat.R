# Seurat label transfer comparator.
#
# Builds a Seurat reference object from 68K raw counts + coarse labels
# (pulled from the pre-processed h5ad via a tiny Python helper), then transfers
# labels to 8K and 3K query datasets using FindTransferAnchors / TransferData.
#
# Pre-requisites:
#   - data/raw/pbmc68k/filtered_matrices_mex/hg19/           (10x MEX)
#   - data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38/      (10x MEX)
#   - data/raw/pbmc3k/mex/                                   (10x MEX, exported by src/export_3k_to_mex.py)
#   - results/comparators/seurat/labels/pbmc68k_coarse_labels.csv   (barcode,label)
#   - results/comparators/seurat/labels/pbmc8k_labels.csv
#   - results/comparators/seurat/labels/pbmc3k_labels.csv
# The label CSV files are written by src/export_labels_for_seurat.py.
#
# Usage:
#   python src/export_labels_for_seurat.py
#   Rscript src/comparator_seurat.R

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

RAW_68K_DIR <- "data/raw/pbmc68k/filtered_matrices_mex/hg19"
RAW_8K_DIR  <- "data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38"
RAW_3K_DIR  <- "data/raw/pbmc3k/mex"

LABELS_68K <- "results/comparators/seurat/labels/pbmc68k_coarse_labels.csv"
LABELS_8K  <- "results/comparators/seurat/labels/pbmc8k_labels.csv"
LABELS_3K  <- "results/comparators/seurat/labels/pbmc3k_labels.csv"

OUTDIR <- "results/comparators/seurat"
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

# ══════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════
read_10x_mex <- function(mex_dir) {
  mat_file    <- file.path(mex_dir, "matrix.mtx")
  bc_file     <- file.path(mex_dir, "barcodes.tsv")
  gene_file   <- file.path(mex_dir, "genes.tsv")
  if (!file.exists(gene_file)) gene_file <- file.path(mex_dir, "features.tsv")
  if (!file.exists(mat_file))  mat_file  <- paste0(mat_file, ".gz")
  if (!file.exists(bc_file))   bc_file   <- paste0(bc_file, ".gz")
  if (!file.exists(gene_file)) gene_file <- paste0(gene_file, ".gz")

  mat <- readMM(mat_file)
  bcs <- readLines(bc_file)
  gen <- read.delim(gene_file, header = FALSE, stringsAsFactors = FALSE)
  rownames(mat) <- make.unique(gen[, 2])
  colnames(mat) <- bcs
  mat
}

build_seurat <- function(counts, labels_df = NULL, project_name = "pbmc") {
  obj <- CreateSeuratObject(counts = counts, project = project_name,
                             min.cells = 3, min.features = 200)
  if (!is.null(labels_df)) {
    # Align barcodes to labels
    shared <- intersect(colnames(obj), labels_df$barcode)
    obj <- subset(obj, cells = shared)
    labs <- labels_df$label[match(colnames(obj), labels_df$barcode)]
    obj$coarse_label <- factor(labs)
  }
  obj
}

normalize_pipeline <- function(obj) {
  obj <- NormalizeData(obj, verbose = FALSE)
  obj <- FindVariableFeatures(obj, selection.method = "vst",
                               nfeatures = 2000, verbose = FALSE)
  obj <- ScaleData(obj, verbose = FALSE)
  obj <- RunPCA(obj, npcs = 30, verbose = FALSE)
  obj
}

# ══════════════════════════════════════════════════
# BUILD 68K REFERENCE
# ══════════════════════════════════════════════════
cat("\n── Building 68K Seurat reference ──\n")
cat("  loading raw counts…\n")
counts_68k <- read_10x_mex(RAW_68K_DIR)
cat("  counts shape:", nrow(counts_68k), "×", ncol(counts_68k), "\n")

cat("  loading labels…\n")
labels_68k <- read.csv(LABELS_68K, stringsAsFactors = FALSE)
cat("  label rows:", nrow(labels_68k), "\n")

ref <- build_seurat(counts_68k, labels_df = labels_68k, project_name = "pbmc68k")
cat("  after QC + label alignment:", ncol(ref), "cells\n")

cat("  normalising + PCA…\n")
t0 <- Sys.time()
ref <- normalize_pipeline(ref)
cat(sprintf("  done in %.1f min\n", as.numeric(Sys.time() - t0, units = "mins")))

# Save reference
saveRDS(ref, file.path(OUTDIR, "pbmc68k_seurat_reference.rds"))

# ══════════════════════════════════════════════════
# TRANSFER TO 8K AND 3K
# ══════════════════════════════════════════════════
run_transfer <- function(ref, query_counts, labels_df, ds_name) {
  cat(sprintf("\n── Label transfer → %s ──\n", ds_name))
  cat("  building query Seurat object…\n")
  query <- build_seurat(query_counts, labels_df = labels_df, project_name = ds_name)
  query <- NormalizeData(query, verbose = FALSE)
  query <- FindVariableFeatures(query, selection.method = "vst",
                                  nfeatures = 2000, verbose = FALSE)

  cat("  FindTransferAnchors…\n")
  t0 <- Sys.time()
  anchors <- FindTransferAnchors(
    reference = ref, query = query,
    dims = 1:30, reference.reduction = "pca",
    verbose = FALSE
  )
  cat(sprintf("  anchors found in %.1f min\n",
              as.numeric(Sys.time() - t0, units = "mins")))

  cat("  TransferData…\n")
  t0 <- Sys.time()
  preds <- TransferData(
    anchorset = anchors,
    refdata = ref$coarse_label,
    dims = 1:30,
    verbose = FALSE
  )
  elapsed <- as.numeric(Sys.time() - t0, units = "mins")
  cat(sprintf("  transferred in %.1f min\n", elapsed))

  df <- data.frame(
    barcode         = colnames(query),
    predicted_label = preds$predicted.id,
    prediction_score = apply(as.matrix(preds[, grep("prediction.score.", colnames(preds))]),
                              1, max),
    stringsAsFactors = FALSE
  )
  out_file <- file.path(OUTDIR, paste0("seurat_", tolower(ds_name), "_predictions.csv"))
  write.csv(df, out_file, row.names = FALSE)
  cat("  saved:", out_file, "\n")

  list(elapsed_min = elapsed, n_cells = nrow(df))
}

counts_8k <- read_10x_mex(RAW_8K_DIR)
labels_8k <- read.csv(LABELS_8K, stringsAsFactors = FALSE)
info_8k <- run_transfer(ref, counts_8k, labels_8k, "8K")

counts_3k <- read_10x_mex(RAW_3K_DIR)
labels_3k <- read.csv(LABELS_3K, stringsAsFactors = FALSE)
info_3k <- run_transfer(ref, counts_3k, labels_3k, "3K")

# ══════════════════════════════════════════════════
# RUN INFO
# ══════════════════════════════════════════════════
run_info <- data.frame(
  dataset     = c("8K", "3K"),
  n_cells     = c(info_8k$n_cells, info_3k$n_cells),
  runtime_min = c(info_8k$elapsed_min, info_3k$elapsed_min),
  tool        = "Seurat_label_transfer",
  seurat_version = as.character(packageVersion("Seurat")),
  r_version      = paste(R.version$major, R.version$minor, sep = ".")
)
write.csv(run_info, file.path(OUTDIR, "seurat_run_info.csv"), row.names = FALSE)

cat("\n✅ Seurat label transfer complete.\n")
cat("   Predictions saved to:", OUTDIR, "\n")
cat("   Run metrics post-processing next:\n")
cat("     python src/comparator_seurat_postprocess.py\n")
