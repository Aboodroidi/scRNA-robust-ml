"""Shared fixtures for scRNA-robust-ml tests."""
import sys
import os
import numpy as np
import pytest
import torch

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))


@pytest.fixture
def rng():
    """Seeded NumPy random generator for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def sparse_expression_matrix(rng):
    """Synthetic expression matrix: 100 cells x 50 genes, ~70 % zeros."""
    mat = rng.rand(100, 50).astype(np.float32)
    mat[mat < 0.7] = 0.0
    return mat


@pytest.fixture
def scaled_expression_matrix(sparse_expression_matrix):
    """Standardised version of the sparse expression matrix."""
    mean = sparse_expression_matrix.mean(axis=0, keepdims=True)
    std = sparse_expression_matrix.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((sparse_expression_matrix - mean) / std).astype(np.float32)


@pytest.fixture
def labels_3class(rng):
    """100 integer labels in {0, 1, 2} (stratified-friendly)."""
    return rng.choice(3, size=100).astype(int)
