"""Tests for sparsity mask generation (build_sann_input)."""
import numpy as np
import pytest
from train_all_full_models import build_sann_input


class TestBuildSannInput:
    """Tests for build_sann_input()."""

    def test_output_shape_doubles_features(self, scaled_expression_matrix, sparse_expression_matrix):
        """Output should have 2× the number of gene columns (expression + mask)."""
        result = build_sann_input(scaled_expression_matrix, sparse_expression_matrix)
        n_cells, n_genes = sparse_expression_matrix.shape
        assert result.shape == (n_cells, 2 * n_genes)

    def test_mask_is_binary(self, scaled_expression_matrix, sparse_expression_matrix):
        """The mask half should contain only 0.0 and 1.0."""
        result = build_sann_input(scaled_expression_matrix, sparse_expression_matrix)
        n_genes = sparse_expression_matrix.shape[1]
        mask_part = result[:, n_genes:]
        unique = np.unique(mask_part)
        assert set(unique).issubset({0.0, 1.0})

    def test_mask_matches_nonzero_pattern(self, scaled_expression_matrix, sparse_expression_matrix):
        """Mask should be 1 where raw expression != 0 and 0 otherwise."""
        result = build_sann_input(scaled_expression_matrix, sparse_expression_matrix)
        n_genes = sparse_expression_matrix.shape[1]
        mask_part = result[:, n_genes:]
        expected_mask = (sparse_expression_matrix != 0).astype(np.float32)
        np.testing.assert_array_equal(mask_part, expected_mask)

    def test_expression_half_preserved(self, scaled_expression_matrix, sparse_expression_matrix):
        """The first half of the output should equal the scaled expression input."""
        result = build_sann_input(scaled_expression_matrix, sparse_expression_matrix)
        n_genes = sparse_expression_matrix.shape[1]
        expr_part = result[:, :n_genes]
        np.testing.assert_array_almost_equal(expr_part, scaled_expression_matrix)

    def test_dtype_is_float32(self, scaled_expression_matrix, sparse_expression_matrix):
        """Output must be float32."""
        result = build_sann_input(scaled_expression_matrix, sparse_expression_matrix)
        assert result.dtype == np.float32

    def test_all_zeros_raw_gives_zero_mask(self):
        """If raw expression is all zeros, mask should be all zeros."""
        scaled = np.ones((5, 10), dtype=np.float32)
        raw = np.zeros((5, 10), dtype=np.float32)
        result = build_sann_input(scaled, raw)
        mask_part = result[:, 10:]
        assert mask_part.sum() == 0.0

    def test_all_nonzero_raw_gives_ones_mask(self):
        """If raw expression has no zeros, mask should be all ones."""
        scaled = np.zeros((5, 10), dtype=np.float32)
        raw = np.ones((5, 10), dtype=np.float32)
        result = build_sann_input(scaled, raw)
        mask_part = result[:, 10:]
        assert mask_part.sum() == 5 * 10
