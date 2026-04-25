#!/usr/bin/env python
"""
Train PCA-based LR / XGBoost / SANN models on PBMC 8K, evaluate externally
on PBMC 3K and PBMC 68K.

Mirror of train_3k_eval_8k_68k.py with 8K as the training source. Imports
model, training, and eval helpers from that file to avoid duplication.

Notes:
  - 8K's preprocessed h5ad is already aligned to 68K's HVG set (2000 genes)
    by preprocess_pbmc8k.py, so 68K evaluation is in the natural gene
    space (100% overlap by construction).
  - preprocess_pbmc8k.py does NOT save per-gene mean/std, so we reload
    raw 8K (mtx → QC → CPM → log1p → align to 68K HVGs) and compute the
    z-score stats on the fly from the same cells used for training.
  - --gene_intersection (default ON) further restricts to genes also
    present in 3K so 3K external evaluation avoids zero-fill bias.
  - All regularization tricks inherited from the 3K script: balanced
    sampling for SANN, class_weight='balanced' for LR, sample weights for
    XGB, optional --prior_correction at inference.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import joblib

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

import xgboost as xgb

# Reuse building blocks from the 3K training script
from train_3k_eval_8k_68k import (
    train_lr, train_xgb, train_sann, predict_sann,
    evaluate_external, align_and_zscore,
    reconstruct_68k_lognorm, load_8k_lognorm,
    to_dense_f32,
)


# ============================================================
# 8K training-space loader
# ============================================================
def load_8k_training_space(raw_8k_dir, data_8k_labeled):
    """
    Load 8K labels + reload raw 8K counts to produce:
      - log-norm expression aligned to 68K HVGs (the 8K feature space)
      - per-gene mean/std (8K's z-score stats)
      - z-scored expression (what models train on)
      - labeled AnnData restricted to cells present in raw-QC survivors
    """
    print(f"Loading 8K labels: {data_8k_labeled}")
    ad8 = sc.read_h5ad(data_8k_labeled)
    genes_8k = list(ad8.var_names)
    print(f"  8K labeled: {ad8.shape} ({len(genes_8k)} genes = 68K HVGs)")

    print("Reloading raw 8K for log-norm + stats ...")
    X_log_raw, genes_raw, obs_raw = load_8k_lognorm(raw_8k_dir)

    # Align raw log-norm to the 8K labeled gene order (= 68K HVG order)
    src_idx = {g: i for i, g in enumerate(genes_raw)}
    X_log = np.zeros((X_log_raw.shape[0], len(genes_8k)), dtype=np.float32)
    shared = 0
    for i, g in enumerate(genes_8k):
        j = src_idx.get(g)
        if j is not None:
            X_log[:, i] = X_log_raw[:, j]
            shared += 1
    print(f"  raw→labeled gene alignment: {shared}/{len(genes_8k)}")

    # Keep only cells present in BOTH raw reload and labeled file
    raw_idx = {n: i for i, n in enumerate(obs_raw)}
    kept = [n for n in ad8.obs_names if n in raw_idx]
    if len(kept) < ad8.shape[0]:
        print(f"  WARNING: {ad8.shape[0] - len(kept)} labeled cells dropped "
              "(not present in raw-QC survivors)")
    keep_rows = np.array([raw_idx[n] for n in kept])
    X_log = X_log[keep_rows]
    ad8 = ad8[kept].copy()

    # Compute 8K's per-gene stats on the training-space log-norm data
    ref_mean = X_log.mean(axis=0).astype(np.float32)
    ref_std = X_log.std(axis=0).astype(np.float32)
    std_safe = np.where(ref_std == 0, 1.0, ref_std)
    X_z = np.clip((X_log - ref_mean) / std_safe, -10, 10).astype(np.float32)

    return ad8, X_z, X_log, genes_8k, ref_mean, ref_std


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Train on PBMC 8K, eval externally on 3K + 68K"
    )
    ap.add_argument("--data_8k", default="data/processed/pbmc8k_labeled.h5ad")
    ap.add_argument("--raw_8k_dir",
                    default="data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38")
    ap.add_argument("--data_3k", default="data/processed/pbmc3k_labeled.h5ad")
    ap.add_argument("--data_68k", default="data/processed/pbmc68k_labeled.h5ad")
    ap.add_argument("--outdir", default="results/train_8k")
    ap.add_argument("--label_key", type=str, default="cell_type")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--mask_pca_dim", type=int, default=50)
    ap.add_argument("--models", type=str, default="lr,xgb,sann")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_sann_seeds", type=int, default=3)
    ap.add_argument("--sann_arch", type=str, default="small",
                    choices=["small", "big"])
    ap.add_argument("--sann_batch_size", type=int, default=256)
    ap.add_argument("--sann_patience", type=int, default=50)
    ap.add_argument("--sann_mixup_alpha", type=float, default=0.4)
    ap.add_argument("--no_intersection", action="store_true",
                    help="Disable 8K ∩ 3K gene-intersection filter "
                         "(default: intersection is ON).")
    ap.add_argument("--no_coarsen", action="store_true",
                    help="Disable label coarsening. Default: ON. Coarsening "
                         "merges CD8 T→T cells, Plasma→B cells, and drops "
                         "CL X cells so the class space matches 3K/68K.")
    ap.add_argument("--prior_correction", action="store_true")
    args = ap.parse_args()
    args.models = [m.strip().lower() for m in args.models.split(",")]
    use_intersection = not args.no_intersection
    use_coarsen = not args.no_coarsen

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load 8K training data (labels + log-norm + stats)
    # ------------------------------------------------------------------
    (ad8, X_8k_z_full, X_8k_log_full, genes_8k_full,
     ref_mean_full, ref_std_full) = load_8k_training_space(
        args.raw_8k_dir, args.data_8k
    )

    # ------------------------------------------------------------------
    # 1b. (Option A) Intersect 8K genes with 3K gene set
    # ------------------------------------------------------------------
    if use_intersection:
        print(f"\n[Option A] Intersecting 8K genes ({len(genes_8k_full)}) "
              f"with 3K gene set ...")
        ad3_header = sc.read_h5ad(args.data_3k, backed="r")
        genes_3k_set = set(ad3_header.var_names)
        ad3_header.file.close()
        keep_mask = np.array([g in genes_3k_set for g in genes_8k_full])
        n_kept = int(keep_mask.sum())
        print(f"  8K ∩ 3K: {n_kept}/{len(genes_8k_full)} genes retained")
        if n_kept < 500:
            raise RuntimeError(f"Too few shared genes ({n_kept}) — "
                               "check gene-name conventions.")
        genes_8k = [g for g, k in zip(genes_8k_full, keep_mask) if k]
        ref_mean = ref_mean_full[keep_mask]
        ref_std = ref_std_full[keep_mask]
        X_8k_z = X_8k_z_full[:, keep_mask]
        X_8k_log = X_8k_log_full[:, keep_mask]
        with open(os.path.join(args.outdir, "intersection_genes.txt"), "w") as f:
            f.write("\n".join(genes_8k))
    else:
        print("\n[Option A DISABLED] Using all 8K (= 68K HVG) genes — "
              "expect 3K external eval to suffer from zero-fill bias.")
        genes_8k = genes_8k_full
        ref_mean = ref_mean_full
        ref_std = ref_std_full
        X_8k_z = X_8k_z_full
        X_8k_log = X_8k_log_full
    ref_std_safe = np.where(ref_std == 0, 1.0, ref_std)

    # ------------------------------------------------------------------
    # 2. Labels + (optional) coarsening + stratified split
    # ------------------------------------------------------------------
    raw_labels = ad8.obs[args.label_key].astype(str).to_numpy()
    print(f"\n  raw 8K label counts: "
          f"{dict(zip(*np.unique(raw_labels, return_counts=True)))}")

    if use_coarsen:
        COARSE_MAP = {
            "T cells": "T cells",
            "CD8 T": "T cells",
            "CD4 T": "T cells",
            "Treg": "T cells",
            "NK": "NK",
            "NK cells": "NK",
            "B cells": "B cells",
            "B": "B cells",
            "Plasma": "B cells",
            "Mono": "Mono",
            "Monocytes": "Mono",
            "CD14+ Monocytes": "Mono",
            "FCGR3A+ Monocytes": "Mono",
            "DC": "DC",
            "Dendritic cells": "DC",
            "Platelet": "Platelet",
            "Megakaryocytes": "Platelet",
        }
        mapped = np.array([COARSE_MAP.get(n, None) for n in raw_labels])
        keep = np.array([m is not None for m in mapped])
        n_dropped = int((~keep).sum())
        dropped_types = sorted(set(raw_labels[~keep].tolist())) if n_dropped else []
        print(f"  [coarsen] dropping {n_dropped} cells with unmapped labels: "
              f"{dropped_types}")

        X_8k_z = X_8k_z[keep]
        X_8k_log = X_8k_log[keep]
        mapped = mapped[keep].astype(str)

        # Build categorical codes in sorted-name order for stable indexing
        class_names = sorted(set(mapped.tolist()))
        name_to_idx = {n: i for i, n in enumerate(class_names)}
        y_8k = np.array([name_to_idx[n] for n in mapped], dtype=np.int64)
        num_classes = len(class_names)
        print(f"  [coarsen] classes ({num_classes}): {class_names}")
        print(f"  [coarsen] counts: "
              f"{dict(zip(class_names, np.bincount(y_8k).tolist()))}")
    else:
        y_cat = ad8.obs[args.label_key].astype("category")
        y_8k = y_cat.cat.codes.to_numpy()
        class_names = list(y_cat.cat.categories)
        num_classes = len(class_names)
        print(f"  classes ({num_classes}): {class_names}")
        print(f"  counts: "
              f"{dict(zip(class_names, np.bincount(y_8k).tolist()))}")

    # Drop singleton classes so stratified split and class weighting work
    counts_check = np.bincount(y_8k, minlength=num_classes)
    singleton = [i for i, c in enumerate(counts_check) if c < 2]
    if singleton:
        drop_names = [class_names[i] for i in singleton]
        print(f"  WARNING: dropping singleton classes: {drop_names}")
        keep_y_mask = ~np.isin(y_8k, singleton)
        y_8k = y_8k[keep_y_mask]
        X_8k_z = X_8k_z[keep_y_mask]
        X_8k_log = X_8k_log[keep_y_mask]
        # Relabel to a contiguous class space for downstream code
        unique_classes = sorted(set(y_8k.tolist()))
        remap = {old: new for new, old in enumerate(unique_classes)}
        y_8k = np.array([remap[c] for c in y_8k], dtype=np.int64)
        class_names = [class_names[i] for i in unique_classes]
        num_classes = len(class_names)
        print(f"  after drop: classes={class_names}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac,
                                 random_state=args.seed)
    tr_idx, val_idx = next(sss.split(np.zeros(len(y_8k)), y_8k))
    y_tr = y_8k[tr_idx]
    y_val = y_8k[val_idx]
    print(f"  train={len(tr_idx)} val={len(val_idx)}")

    X_tr_z = X_8k_z[tr_idx]
    X_val_z = X_8k_z[val_idx]
    X_tr_log = X_8k_log[tr_idx]
    X_val_log = X_8k_log[val_idx]

    # ------------------------------------------------------------------
    # 3. PCA (expression + mask) on train split only
    # ------------------------------------------------------------------
    print("\nFitting expression PCA on train split...")
    expr_pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    X_tr_pca = expr_pca.fit_transform(X_tr_z).astype(np.float32)
    X_val_pca = expr_pca.transform(X_val_z).astype(np.float32)
    print(f"  expr PCA expl. var. sum = "
          f"{expr_pca.explained_variance_ratio_.sum():.4f}")
    joblib.dump(expr_pca, os.path.join(args.outdir, "expr_pca_model.pkl"))

    print("\nFitting mask PCA on train split...")
    X_tr_mask = (X_tr_log != 0).astype(np.float32)
    X_val_mask = (X_val_log != 0).astype(np.float32)
    print(f"  train mask non-zero fraction: {X_tr_mask.mean():.4f}")
    mask_pca = PCA(n_components=args.mask_pca_dim, random_state=args.seed)
    X_tr_mpca = mask_pca.fit_transform(X_tr_mask).astype(np.float32)
    X_val_mpca = mask_pca.transform(X_val_mask).astype(np.float32)
    print(f"  mask PCA expl. var. sum = "
          f"{mask_pca.explained_variance_ratio_.sum():.4f}")
    joblib.dump(mask_pca, os.path.join(args.outdir, "mask_pca_model.pkl"))

    X_tr_full = np.concatenate([X_tr_pca, X_tr_mpca], axis=1).astype(np.float32)
    X_val_full = np.concatenate([X_val_pca, X_val_mpca], axis=1).astype(np.float32)
    print(f"  LR/XGB input: {X_tr_pca.shape}, SANN input: {X_tr_full.shape}")

    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({
            "train_source": "pbmc8k",
            "gene_intersection_with_3k": bool(use_intersection),
            "coarsen_labels": bool(use_coarsen),
            "n_genes_8k_original": int(len(genes_8k_full)),
            "n_genes_used": int(len(genes_8k)),
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
            "pca_dim": int(args.pca_dim),
            "mask_pca_dim": int(args.mask_pca_dim),
            "class_names": class_names,
            "label_key": args.label_key,
            "expr_pca_expl_var": float(expr_pca.explained_variance_ratio_.sum()),
            "mask_pca_expl_var": float(mask_pca.explained_variance_ratio_.sum()),
            "sann_arch": args.sann_arch,
            "sann_n_seeds": int(args.n_sann_seeds),
            "sann_batch_size": int(args.sann_batch_size),
            "sann_patience": int(args.sann_patience),
            "sann_mixup_alpha": float(args.sann_mixup_alpha),
            "prior_correction": bool(args.prior_correction),
            "lr_class_weight": "balanced",
            "xgb_sample_weight": "balanced (1 / class_count)",
            "sann_sampler": "WeightedRandomSampler (class-balanced)",
        }, f, indent=2)

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    train_summary = []
    if "lr" in args.models:
        train_summary.append(train_lr(X_tr_pca, y_tr, X_val_pca, y_val,
                                      args.outdir))
    if "xgb" in args.models:
        train_summary.append(train_xgb(X_tr_pca, y_tr, X_val_pca, y_val,
                                       num_classes, args.outdir))
    if "sann" in args.models:
        stale = os.path.join(args.outdir, "sann_model.pt")
        if os.path.exists(stale):
            os.remove(stale)
        for i in range(args.n_sann_seeds):
            seed_i = args.seed + i
            train_summary.append(train_sann(
                X_tr_full, y_tr, X_val_full, y_val, num_classes,
                args.outdir, n_expr_features=args.pca_dim, seed=seed_i,
                arch=args.sann_arch,
                batch_size=args.sann_batch_size,
                patience=args.sann_patience,
                mixup_alpha=args.sann_mixup_alpha,
                model_filename=f"sann_model_seed{i}.pt",
            ))
    with open(os.path.join(args.outdir, "train_summary.json"), "w") as f:
        json.dump(train_summary, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # 5. Prepare 3K external set (align → z-score with 8K stats → project)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Preparing 3K external set")
    print("=" * 60)
    ad3 = sc.read_h5ad(args.data_3k)
    if "log_norm" in ad3.layers:
        X_3k_log_raw = to_dense_f32(ad3.layers["log_norm"])
    else:
        # fallback: invert 3K's own z-score using 3K stats
        m3 = ad3.var["mean"].values.astype(np.float32)
        s3 = ad3.var["std"].values.astype(np.float32)
        s3 = np.where(s3 == 0, 1.0, s3)
        X_3k_log_raw = to_dense_f32(ad3.X) * s3 + m3
    genes_3k_list = list(ad3.var_names)

    X_3k_log_ref, X_3k_z_ref, shared_3k = align_and_zscore(
        X_3k_log_raw, genes_3k_list, genes_8k, ref_mean, ref_std_safe
    )
    print(f"  3K aligned to 8K genes: {X_3k_z_ref.shape}, "
          f"{shared_3k}/{len(genes_8k)} shared")

    X_3k_epca = expr_pca.transform(X_3k_z_ref).astype(np.float32)
    X_3k_mask = (X_3k_log_ref != 0).astype(np.float32)
    X_3k_mpca = mask_pca.transform(X_3k_mask).astype(np.float32)
    X_3k_sann = np.concatenate([X_3k_epca, X_3k_mpca], axis=1).astype(np.float32)

    # 3K label → 8K class-space mapping
    # 3K classes: T cells, NK, B cells, Mono, DC, Platelet
    # 8K may have both "T cells" and "CD8 T" (or only one). Try both aliases.
    y_3k_cat = ad3.obs["cell_type_coarse"].astype(str)
    LABEL_ALIAS_3K = {
        "T cells": ["T cells", "CD8 T"],
        "NK": ["NK"],
        "B cells": ["B cells"],
        "Mono": ["Mono"],
        "DC": ["DC"],
        "Platelet": ["Platelet"],
    }

    def resolve_3k(name):
        if name in class_names:
            return class_names.index(name)
        for alias in LABEL_ALIAS_3K.get(name, []):
            if alias in class_names:
                return class_names.index(alias)
        return -1

    label_map_3k = {name: resolve_3k(name) for name in y_3k_cat.unique()}
    print(f"  3K label map: {label_map_3k}")
    y_3k_mapped = np.array([label_map_3k[n] for n in y_3k_cat])
    known_3k = y_3k_mapped >= 0
    known_cls_3k = sorted(set(int(v) for v in y_3k_mapped[known_3k].tolist()))
    print(f"  3K cells: {len(y_3k_mapped)} total, {known_3k.sum()} known")
    unk_3k = [n for n in y_3k_cat.unique() if label_map_3k[n] == -1]
    if unk_3k:
        print(f"  3K types not in 8K class space (excluded): {unk_3k}")

    # ------------------------------------------------------------------
    # 6. Prepare 68K external set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Preparing 68K external set")
    print("=" * 60)
    ad68 = sc.read_h5ad(args.data_68k)
    X_68k_log_raw, genes_68k = reconstruct_68k_lognorm(ad68)
    X_68k_log_ref, X_68k_z_ref, shared_68k = align_and_zscore(
        X_68k_log_raw, genes_68k, genes_8k, ref_mean, ref_std_safe
    )
    print(f"  68K aligned to 8K genes: {X_68k_z_ref.shape}, "
          f"{shared_68k}/{len(genes_8k)} shared")

    X_68k_epca = expr_pca.transform(X_68k_z_ref).astype(np.float32)
    X_68k_mask = (X_68k_log_ref != 0).astype(np.float32)
    X_68k_mpca = mask_pca.transform(X_68k_mask).astype(np.float32)
    X_68k_sann = np.concatenate([X_68k_epca, X_68k_mpca],
                                axis=1).astype(np.float32)

    y_68k_cat = ad68.obs["cell_type"].astype(str)
    class_names_68k = sorted(y_68k_cat.unique())
    print(f"  68K classes ({len(class_names_68k)}): {class_names_68k}")

    LABEL_ALIAS_68K = {
        "CD8 T": ["CD8 T", "T cells"],
        "CD4 T": ["T cells", "CD8 T"],
        "Treg": ["T cells"],
        "NK cells": ["NK"],
        "NK": ["NK"],
        "B cells": ["B cells"],
        "B": ["B cells"],
        "Plasma": ["B cells"],
        "Mono": ["Mono"],
        "Monocytes": ["Mono"],
        "CD14+ Monocytes": ["Mono"],
        "FCGR3A+ Monocytes": ["Mono"],
        "DC": ["DC"],
        "Dendritic cells": ["DC"],
        "Platelet": ["Platelet"],
        "Megakaryocytes": ["Platelet"],
    }

    def resolve_68k(name):
        if name in class_names:
            return class_names.index(name)
        for alias in LABEL_ALIAS_68K.get(name, []):
            if alias in class_names:
                return class_names.index(alias)
        n = str(name)
        if ("T cell" in n or "CD4" in n or "CD8" in n):
            for a in ("T cells", "CD8 T"):
                if a in class_names:
                    return class_names.index(a)
        if "NK" in n and "NK" in class_names:
            return class_names.index("NK")
        if ("B cell" in n) and "B cells" in class_names:
            return class_names.index("B cells")
        if "Mono" in n and "Mono" in class_names:
            return class_names.index("Mono")
        if "DC" in n and "DC" in class_names:
            return class_names.index("DC")
        if ("Platelet" in n or "Megakary" in n) and "Platelet" in class_names:
            return class_names.index("Platelet")
        return -1

    label_map_68k = {name: resolve_68k(name) for name in class_names_68k}
    print(f"  68K label map: {label_map_68k}")
    y_68k_mapped = np.array([label_map_68k[n] for n in y_68k_cat])
    known_68k = y_68k_mapped >= 0
    known_cls_68k = sorted(set(int(v) for v in y_68k_mapped[known_68k].tolist()))
    print(f"  68K cells: {len(y_68k_mapped)} total, {known_68k.sum()} known")
    unk_68k = [n for n in class_names_68k if label_map_68k[n] == -1]
    if unk_68k:
        print(f"  68K types not in 8K class space (excluded): {unk_68k}")

    # ------------------------------------------------------------------
    # 7. Predict + evaluate
    # ------------------------------------------------------------------
    external_rows = []

    train_prior = np.bincount(y_tr, minlength=num_classes).astype(np.float64)
    train_prior = np.maximum(train_prior, 1.0) / train_prior.sum()
    np.save(os.path.join(args.outdir, "train_prior.npy"), train_prior)
    if args.prior_correction:
        print(f"\n[Prior correction] train prior: "
              f"{dict(zip(class_names, np.round(train_prior, 4)))}")

    def _apply_prior(probs):
        if not args.prior_correction:
            return probs
        adj = probs / train_prior[None, :]
        adj = adj / adj.sum(axis=1, keepdims=True)
        return adj.astype(np.float32)

    def _run_lr(X_pca_feats, y_mapped, known_mask, known_cls, tag):
        lr_model = joblib.load(os.path.join(args.outdir, "lr_model.pkl"))
        lr_scaler = joblib.load(os.path.join(args.outdir, "lr_scaler.pkl"))
        probs = lr_model.predict_proba(lr_scaler.transform(X_pca_feats))
        probs = _apply_prior(probs)
        pred = probs.argmax(axis=1)
        return evaluate_external(
            "LR", y_mapped[known_mask], pred[known_mask],
            probs[known_mask], class_names, args.outdir, tag,
            known_class_indices=known_cls,
        )

    def _run_xgb(X_pca_feats, y_mapped, known_mask, known_cls, tag):
        booster = xgb.Booster()
        booster.load_model(os.path.join(args.outdir, "xgb_model.json"))
        probs = booster.predict(xgb.DMatrix(X_pca_feats))
        probs = _apply_prior(probs)
        pred = probs.argmax(axis=1)
        return evaluate_external(
            "XGB", y_mapped[known_mask], pred[known_mask],
            probs[known_mask], class_names, args.outdir, tag,
            known_class_indices=known_cls,
        )

    def _run_sann(X_sann_feats, y_mapped, known_mask, known_cls, tag):
        probs = predict_sann(args.outdir, args.pca_dim, args.mask_pca_dim,
                             num_classes, X_sann_feats, arch=args.sann_arch)
        probs = _apply_prior(probs)
        pred = probs.argmax(axis=1)
        return evaluate_external(
            "SANN", y_mapped[known_mask], pred[known_mask],
            probs[known_mask], class_names, args.outdir, tag,
            known_class_indices=known_cls,
        )

    if "lr" in args.models:
        external_rows.append(_run_lr(X_3k_epca, y_3k_mapped,
                                     known_3k, known_cls_3k, "pbmc3k"))
        external_rows.append(_run_lr(X_68k_epca, y_68k_mapped,
                                     known_68k, known_cls_68k, "pbmc68k"))
    if "xgb" in args.models:
        external_rows.append(_run_xgb(X_3k_epca, y_3k_mapped,
                                      known_3k, known_cls_3k, "pbmc3k"))
        external_rows.append(_run_xgb(X_68k_epca, y_68k_mapped,
                                      known_68k, known_cls_68k, "pbmc68k"))
    if "sann" in args.models:
        external_rows.append(_run_sann(X_3k_sann, y_3k_mapped,
                                       known_3k, known_cls_3k, "pbmc3k"))
        external_rows.append(_run_sann(X_68k_sann, y_68k_mapped,
                                       known_68k, known_cls_68k, "pbmc68k"))

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    summary = pd.DataFrame(external_rows)
    summary_path = os.path.join(args.outdir, "external_eval_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\n" + "=" * 60)
    print("  SUMMARY — 8K-trained models on external test sets")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Artifacts: {args.outdir}/")


if __name__ == "__main__":
    main()
