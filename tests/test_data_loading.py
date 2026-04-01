"""Tests for data loading, splitting, and standardisation utilities."""
import json
import os
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from train_all_full_models import (
    build_sann_input,
    load_expression_data,
    load_fixed_split,
    load_labels,
    make_train_val_split,
    standardize_expression_train_val_test,
    to_dense_float32,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_adata(n_cells=80, n_genes=30, n_hvg=15, sparse=False):
    """Create a minimal AnnData object for testing."""
    rng = np.random.RandomState(0)
    X = rng.rand(n_cells, n_genes).astype(np.float32)
    if sparse:
        X = sp.csr_matrix(X)

    obs = pd.DataFrame({"cell_type": np.tile(["A", "B", "C", "D"], n_cells // 4)},
                        index=[f"cell_{i}" for i in range(n_cells)])
    hvg_flags = np.zeros(n_genes, dtype=bool)
    hvg_flags[:n_hvg] = True
    var = pd.DataFrame({"highly_variable": hvg_flags},
                        index=[f"gene_{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# to_dense_float32
# ---------------------------------------------------------------------------
class TestToDenseFloat32:
    def test_dense_input(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = to_dense_float32(arr)
        assert result.dtype == np.float32
        assert isinstance(result, np.ndarray)

    def test_sparse_input(self):
        arr = sp.csr_matrix(np.eye(3, dtype=np.float64))
        result = to_dense_float32(arr)
        assert result.dtype == np.float32
        assert not sp.issparse(result)
        np.testing.assert_array_equal(result, np.eye(3, dtype=np.float32))


# ---------------------------------------------------------------------------
# load_labels
# ---------------------------------------------------------------------------
class TestLoadLabels:
    def test_returns_codes_and_names(self):
        adata = _make_adata()
        y, names = load_labels(adata, label_key="cell_type")
        assert len(y) == adata.n_obs
        assert set(names) == {"A", "B", "C", "D"}
        assert y.dtype == np.intp or np.issubdtype(y.dtype, np.integer)

    def test_missing_key_raises(self):
        adata = _make_adata()
        with pytest.raises(ValueError, match="Expected"):
            load_labels(adata, label_key="nonexistent")


# ---------------------------------------------------------------------------
# load_expression_data  (round-trip through .h5ad on disk)
# ---------------------------------------------------------------------------
class TestLoadExpressionData:
    def test_hvg_filtering(self, tmp_path):
        adata = _make_adata(n_genes=30, n_hvg=15)
        path = str(tmp_path / "test.h5ad")
        adata.write_h5ad(path)

        X, y, names = load_expression_data(path)
        assert X.shape == (80, 15), "Should keep only HVG columns"
        assert X.dtype == np.float32

    def test_max_hvgs_truncation(self, tmp_path):
        adata = _make_adata(n_genes=30, n_hvg=15)
        path = str(tmp_path / "test.h5ad")
        adata.write_h5ad(path)

        X, _, _ = load_expression_data(path, max_hvgs=5)
        assert X.shape[1] == 5

    def test_no_hvg_column_uses_all(self, tmp_path):
        adata = _make_adata()
        del adata.var["highly_variable"]
        path = str(tmp_path / "test.h5ad")
        adata.write_h5ad(path)

        X, _, _ = load_expression_data(path)
        assert X.shape[1] == adata.n_vars

    def test_sparse_h5ad(self, tmp_path):
        adata = _make_adata(sparse=True)
        path = str(tmp_path / "sparse.h5ad")
        adata.write_h5ad(path)

        X, _, _ = load_expression_data(path)
        assert isinstance(X, np.ndarray)
        assert X.dtype == np.float32


# ---------------------------------------------------------------------------
# load_fixed_split
# ---------------------------------------------------------------------------
class TestLoadFixedSplit:
    def test_round_trip(self, tmp_path):
        split = {"train_idx": [0, 1, 2], "test_idx": [3, 4]}
        path = str(tmp_path / "split.json")
        with open(path, "w") as f:
            json.dump(split, f)

        train_idx, test_idx = load_fixed_split(path)
        np.testing.assert_array_equal(train_idx, [0, 1, 2])
        np.testing.assert_array_equal(test_idx, [3, 4])
        assert train_idx.dtype == int

    def test_missing_key_raises(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            json.dump({"train_idx": [0]}, f)

        with pytest.raises(ValueError, match="test_idx"):
            load_fixed_split(path)


# ---------------------------------------------------------------------------
# make_train_val_split
# ---------------------------------------------------------------------------
class TestMakeTrainValSplit:
    def test_disjoint_indices(self, labels_3class):
        tr, val = make_train_val_split(labels_3class, val_frac=0.2, seed=0)
        assert len(set(tr) & set(val)) == 0, "Train and val must be disjoint"

    def test_covers_all_samples(self, labels_3class):
        tr, val = make_train_val_split(labels_3class, val_frac=0.2, seed=0)
        assert len(tr) + len(val) == len(labels_3class)

    def test_approximate_fraction(self, labels_3class):
        tr, val = make_train_val_split(labels_3class, val_frac=0.2, seed=0)
        actual_frac = len(val) / len(labels_3class)
        assert 0.1 <= actual_frac <= 0.3

    def test_deterministic(self, labels_3class):
        tr1, val1 = make_train_val_split(labels_3class, seed=7)
        tr2, val2 = make_train_val_split(labels_3class, seed=7)
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(val1, val2)


# ---------------------------------------------------------------------------
# standardize_expression_train_val_test
# ---------------------------------------------------------------------------
class TestStandardize:
    def _splits(self, rng):
        X_train = rng.rand(60, 10).astype(np.float32)
        X_val = rng.rand(20, 10).astype(np.float32)
        X_test = rng.rand(20, 10).astype(np.float32)
        return X_train, X_val, X_test

    def test_train_mean_near_zero(self, rng):
        Xtr, Xv, Xte = self._splits(rng)
        Xtr_s, _, _, _, _ = standardize_expression_train_val_test(Xtr, Xv, Xte)
        np.testing.assert_allclose(Xtr_s.mean(axis=0), 0.0, atol=1e-5)

    def test_train_std_near_one(self, rng):
        Xtr, Xv, Xte = self._splits(rng)
        Xtr_s, _, _, _, _ = standardize_expression_train_val_test(Xtr, Xv, Xte)
        np.testing.assert_allclose(Xtr_s.std(axis=0), 1.0, atol=0.05)

    def test_output_dtype(self, rng):
        Xtr, Xv, Xte = self._splits(rng)
        results = standardize_expression_train_val_test(Xtr, Xv, Xte)
        for arr in results:
            assert arr.dtype == np.float32

    def test_constant_feature_not_nan(self, rng):
        Xtr, Xv, Xte = self._splits(rng)
        Xtr[:, 0] = 5.0  # constant column
        Xtr_s, Xv_s, Xte_s, _, _ = standardize_expression_train_val_test(Xtr, Xv, Xte)
        assert not np.isnan(Xtr_s).any()
        assert not np.isnan(Xv_s).any()
        assert not np.isnan(Xte_s).any()
