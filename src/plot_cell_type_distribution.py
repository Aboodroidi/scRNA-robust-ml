import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
DATA_68K = "data/processed/pbmc68k_labeled.h5ad"
DATA_8K = "data/processed/pbmc8k_labeled.h5ad"
DATA_3K = "data/processed/pbmc3k_labeled.h5ad"

OUT_DIR = "results/figures"
OUT_PATH = os.path.join(OUT_DIR, "cell_type_distribution_all_donors.png")

PUT_CLUSTERS_LAST = True


def get_counts(path, label_key="cell_type"):
    """Load h5ad and return cell-type value counts as a Series."""
    adata = sc.read_h5ad(path)
    if label_key not in adata.obs:
        raise ValueError(f"Expected adata.obs['{label_key}'] in {path}")
    return adata.obs[label_key].astype(str).value_counts()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load counts for each donor (rename "T cells" → "CD8 T" for consistency)
    counts_68k = get_counts(DATA_68K)
    counts_68k.index = counts_68k.index.map(lambda x: "CD8 T" if x == "T cells" else x)
    counts_8k = get_counts(DATA_8K)
    counts_8k.index = counts_8k.index.map(lambda x: "CD8 T" if x == "T cells" else x)
    counts_3k = get_counts(DATA_3K)
    counts_3k.index = counts_3k.index.map(lambda x: "CD8 T" if x == "T cells" else x)

    # Get union of all cell types, ordered by 68K count (descending)
    # Push CL* labels to the end
    all_types_68k = counts_68k.sort_values(ascending=False)
    if PUT_CLUSTERS_LAST:
        is_cl = all_types_68k.index.str.startswith("CL ")
        all_types_68k = pd.concat([all_types_68k[~is_cl], all_types_68k[is_cl]])

    # Exclude CL 1 and DC (only present in external datasets, not used in evaluation)
    EXCLUDE = {"CL 1", "DC"}
    cell_types = [t for t in all_types_68k.index if t not in EXCLUDE]

    # Build aligned arrays (raw counts)
    raw_68k = [int(counts_68k.get(t, 0)) for t in cell_types]
    raw_8k = [int(counts_8k.get(t, 0)) for t in cell_types]
    raw_3k = [int(counts_3k.get(t, 0)) for t in cell_types]

    # Convert to percentages within each donor
    total_68k = sum(raw_68k)
    total_8k = sum(raw_8k)
    total_3k = sum(raw_3k)

    vals_68k = [v / total_68k * 100 for v in raw_68k]
    vals_8k = [v / total_8k * 100 for v in raw_8k]
    vals_3k = [v / total_3k * 100 for v in raw_3k]

    # Plot — grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 5.5))

    x = np.arange(len(cell_types))
    width = 0.27

    bars_68k = ax.bar(x - width, vals_68k, width, label="68K (Donor A)", color="tab:blue", alpha=0.85)
    bars_8k = ax.bar(x, vals_8k, width, label="8K (Donor B)", color="tab:orange", alpha=0.85)
    bars_3k = ax.bar(x + width, vals_3k, width, label="3K (Donor C)", color="tab:green", alpha=0.85)

    ax.set_xlabel("Cell Type", fontsize=12)
    ax.set_ylabel("Proportion of Cells (%)", fontsize=12)
    ax.set_title("Cell-Type Distribution Across All Three PBMC Donors", fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels(cell_types, rotation=40, ha="right", fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print(f"\nSaved {OUT_PATH}")
    print(f"\n68K total: {total_68k:,} cells across {sum(1 for v in vals_68k if v > 0)} types")
    print(f"8K total:  {total_8k:,} cells across {sum(1 for v in vals_8k if v > 0)} types")
    print(f"3K total:  {total_3k:,} cells across {sum(1 for v in vals_3k if v > 0)} types")

    print("\nPer-type breakdown (%):")
    for t, v68, v8, v3 in zip(cell_types, vals_68k, vals_8k, vals_3k):
        print(f"  {t:20s}  68K={v68:5.1f}%  8K={v8:5.1f}%  3K={v3:5.1f}%")


if __name__ == "__main__":
    main()
