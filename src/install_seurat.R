# Install Seurat (CRAN) for label transfer comparator.
# Run: Rscript src/install_seurat.R
#
# Seurat 5.x pulls heavy compiled dependencies. Allow 20–45 min.

repos <- "https://cloud.r-project.org"

# Core Seurat
install.packages("Seurat", repos = repos)

# Seurat sometimes needs these helpers depending on data formats
# (kept minimal — only what we actually use in the workflow)
install.packages(c("Matrix", "dplyr"), repos = repos)

cat("\n\n✅ Seurat installation complete.\n")
cat("   Version:", as.character(packageVersion("Seurat")), "\n")
