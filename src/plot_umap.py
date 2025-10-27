import scanpy as sc
import matplotlib.pyplot as plt

adata = sc.read("data/processed/pbmc68k_prepped.h5ad")
sc.pl.umap(adata, color=None, show=False)
plt.savefig("data/processed/pbmc68k_umap.png", dpi=200, bbox_inches="tight")
print("Saved data/processed/pbmc68k_umap.png")