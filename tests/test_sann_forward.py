"""Tests for SANN model forward pass (both variants)."""
import numpy as np
import pytest
import torch

# Import both SANN variants
from train_sann import SANN as SANNv1
from train_all_full_models import SANN as SANNv2


class TestSANNv1Forward:
    """Tests for the original SANN (train_sann.py) with GELU + L1."""

    @pytest.fixture
    def model(self):
        return SANNv1(input_dim=100, num_classes=5, hidden=64, dropout=0.0)

    def test_output_shape(self, model):
        x = torch.randn(16, 100)
        model.eval()
        out = model(x)
        assert out.shape == (16, 5)

    def test_output_is_logits(self, model):
        """Output should be raw logits (not bounded to [0,1])."""
        x = torch.randn(32, 100)
        model.eval()
        out = model(x)
        assert out.min().item() < 0 or out.max().item() > 1, (
            "Logits should not all be in [0, 1]"
        )

    def test_single_sample(self, model):
        """Forward pass should work with batch size 1."""
        x = torch.randn(1, 100)
        model.eval()
        out = model(x)
        assert out.shape == (1, 5)

    def test_l1_penalty_positive(self, model):
        """L1 penalty should be a positive scalar."""
        penalty = model.l1_penalty()
        assert penalty.dim() == 0
        assert penalty.item() > 0

    def test_gradient_flows(self):
        model = SANNv1(input_dim=20, num_classes=3, hidden=16, dropout=0.0)
        model.train()
        x = torch.randn(8, 20, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert model.fc1.weight.grad is not None

    def test_different_hidden_sizes(self):
        for h in [32, 128, 256]:
            m = SANNv1(input_dim=50, num_classes=4, hidden=h, dropout=0.0)
            m.eval()
            out = m(torch.randn(4, 50))
            assert out.shape == (4, 4)


class TestSANNv2Forward:
    """Tests for the full-models SANN (train_all_full_models.py) with ReLU."""

    @pytest.fixture
    def model(self):
        return SANNv2(input_dim=100, num_classes=5, hidden1=64, hidden2=32, dropout=0.0)

    def test_output_shape(self, model):
        x = torch.randn(16, 100)
        model.eval()
        out = model(x)
        assert out.shape == (16, 5)

    def test_single_sample(self, model):
        x = torch.randn(1, 100)
        model.eval()
        out = model(x)
        assert out.shape == (1, 5)

    def test_no_batchnorm(self):
        """Model should work with batchnorm disabled."""
        m = SANNv2(input_dim=20, num_classes=3, hidden1=16, hidden2=8,
                   dropout=0.0, use_batchnorm=False)
        m.eval()
        out = m(torch.randn(4, 20))
        assert out.shape == (4, 3)

    def test_gradient_flows(self):
        model = SANNv2(input_dim=20, num_classes=3, hidden1=16, hidden2=8, dropout=0.0)
        model.train()
        x = torch.randn(8, 20, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_deterministic_eval(self):
        """Eval mode should give deterministic outputs."""
        model = SANNv2(input_dim=20, num_classes=3, hidden1=16, hidden2=8, dropout=0.3)
        model.eval()
        x = torch.randn(4, 20)
        out1 = model(x)
        out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_sann_input_integration(self):
        """End-to-end: build_sann_input → SANN forward."""
        from train_all_full_models import build_sann_input

        n_genes = 25
        scaled = np.random.randn(8, n_genes).astype(np.float32)
        raw = np.random.rand(8, n_genes).astype(np.float32)
        raw[raw < 0.5] = 0.0

        sann_in = build_sann_input(scaled, raw)
        model = SANNv2(input_dim=n_genes * 2, num_classes=4, hidden1=32, hidden2=16, dropout=0.0)
        model.eval()
        out = model(torch.from_numpy(sann_in))
        assert out.shape == (8, 4)
