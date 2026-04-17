# Install SingleR + reference atlas + I/O helpers
# Run once: Rscript src/install_singleR.R
# Takes ~30-60 minutes (compiles several C++ dependencies)

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

BiocManager::install(c(
  "SingleR",                 # classifier
  "celldex",                 # reference atlases (Human Primary Cell Atlas, Monaco etc.)
  "SingleCellExperiment",    # data container
  "scuttle",                 # normalisation helpers
  "Matrix"                   # sparse matrices
), update = FALSE, ask = FALSE)

# Optional — for reading 10x mtx files
install.packages(c("jsonlite"), repos = "https://cloud.r-project.org")

cat("\n\n✅ SingleR installation complete.\n")
