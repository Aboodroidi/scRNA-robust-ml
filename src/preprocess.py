# src/preprocess.py
import argparse
import os
import scanpy as sc

def main(in_dir, out_path, n_hvg=2000, n_pcs=50, n_neighbors=15, mt_pct=10.0):
    sc.settings.verbosity = 2

    # Load 10x mtx (works with genes.tsv or features.tsv)
    adata = sc.read_10x_mtx(
        in_dir,
        var_names="gene_symbols",  # use gene symbols from genes.tsv
        make_unique=True
    )

    # Basic QC
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    # filter obvious low-quality cells/genes (tune later)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[adata.obs.pct_counts_mt < mt_pct].copy()

    # Normalise & log
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVGs, scale, PCA
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs)

    # Neighbours + UMAP
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    adata.write(out_path)
    print(f"Saved: {out_path}")
    print(adata)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="data/raw/pbmc68k/filtered_feature_bc_matrix")
    p.add_argument("--out", default="data/processed/pbmc68k_prepped.h5ad")
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--n_pcs", type=int, default=50)
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--mt_pct", type=float, default=10.0)
    args = p.parse_args()
    main(args.in_dir, args.out, args.n_hvg, args.n_pcs, args.n_neighbors, args.mt_pct)