# src/make_splits.py
import os
import json
import argparse

import numpy as np
import scanpy as sc
from sklearn.model_selection import StratifiedShuffleSplit


def parse_args():
    p = argparse.ArgumentParser(description="Create repeated stratified train/test splits (indices).")
    p.add_argument("--data", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--label_key", type=str, default="cell_type")
    p.add_argument("--outdir", type=str, default="splits")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--base_seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    adata = sc.read_h5ad(args.data)
    y = adata.obs[args.label_key].astype("category").cat.codes.to_numpy().astype(int)

    # fixed list of seeds for reproducibility
    seeds = [args.base_seed + i * 11 for i in range(args.n_splits)]

    all_splits = []

    for i, seed in enumerate(seeds, start=1):
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=seed
        )
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
        split_id = f"split_{i:02d}"

        record = {
            "split_id": split_id,
            "seed": int(seed),
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
        }
        all_splits.append(record)

        out_path = os.path.join(args.outdir, f"{split_id}.json")
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        print(f"[Saved] {out_path} | train={len(train_idx)} test={len(test_idx)} seed={seed}")

    all_path = os.path.join(args.outdir, "all_splits.json")
    with open(all_path, "w") as f:
        json.dump(all_splits, f, indent=2)
    print(f"[Saved] {all_path} (n_splits={len(all_splits)})")


if __name__ == "__main__":
    main()