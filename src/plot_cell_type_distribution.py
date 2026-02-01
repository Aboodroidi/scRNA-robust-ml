import os
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
IN_PATH = "data/processed/pbmc68k_labeled.h5ad"   # <- uses your annotated dataset
OUT_DIR = "results/figures"
OUT_PATH = os.path.join(OUT_DIR, "cell_type_distribution.png")

# Optional: if you used labels like "CL 0", "CL 12" etc and want to group them last
PUT_CLUSTERS_LAST = True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load labeled data
    adata = sc.read_h5ad(IN_PATH)

    if "cell_type" not in adata.obs:
        raise ValueError("Expected adata.obs['cell_type'] to exist. Run annotate.py first.")

    # Counts per cell type (absolute counts)
    counts = adata.obs["cell_type"].astype(str).value_counts()

    # Recommended order: descending by count
    counts = counts.sort_values(ascending=False)

    # Optional: push CL* labels to the end (useful if you have unknown clusters)
    if PUT_CLUSTERS_LAST:
        is_cl = counts.index.str.startswith("CL ")
        counts = pd.concat([counts[~is_cl], counts[is_cl]])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Single-color bars (matplotlib default blue). No rainbow / per-class colors.
    ax.bar(counts.index, counts.values)

    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Number of Cells")
    ax.set_title("PBMC 68k Cell-Type Distribution")

    # Rotate x labels for readability
    ax.tick_params(axis="x", labelrotation=40)
    ax.set_axisbelow(True)

    # Remove grid clutter (keep clean)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()