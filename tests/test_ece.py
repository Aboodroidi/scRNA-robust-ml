"""Tests for Expected Calibration Error (ECE) calculation."""
import numpy as np
import pytest

from calibrate import ece_score


class TestECEScore:
    """Tests for ece_score()."""

    def test_perfect_calibration_returns_zero(self):
        """If predicted confidence matches accuracy perfectly, ECE ~ 0."""
        # All predictions correct with 100% confidence
        y_true = np.array([0, 1, 2, 0, 1])
        prob = np.eye(3)[y_true]  # one-hot = 100% confidence on correct class
        ece = ece_score(y_true, prob)
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_completely_wrong_high_confidence(self):
        """High confidence but always wrong → ECE should be significant."""
        y_true = np.array([0, 0, 0, 0, 0])
        # Predicts class 1 with 0.95 confidence (not 1.0, which hits a digitize edge)
        prob = np.full((5, 3), 0.025)
        prob[:, 1] = 0.95
        ece = ece_score(y_true, prob)
        assert ece > 0.5

    def test_ece_in_zero_one_range(self):
        """ECE should always be in [0, 1]."""
        rng = np.random.RandomState(99)
        y_true = rng.choice(4, size=200)
        prob = rng.dirichlet(np.ones(4), size=200)
        ece = ece_score(y_true, prob)
        assert 0.0 <= ece <= 1.0

    def test_returns_float(self):
        y_true = np.array([0, 1])
        prob = np.array([[0.8, 0.2], [0.3, 0.7]])
        result = ece_score(y_true, prob)
        assert isinstance(result, float)

    def test_uniform_predictions(self):
        """Uniform predictions (max conf = 1/K) with random labels."""
        n, k = 100, 4
        y_true = np.zeros(n, dtype=int)
        prob = np.full((n, k), 1.0 / k)
        ece = ece_score(y_true, prob)
        # Max confidence = 0.25, accuracy = 1.0 → gap ≈ 0.75
        assert ece > 0.0

    def test_n_bins_parameter(self):
        """Different bin counts should still produce valid ECE."""
        rng = np.random.RandomState(42)
        y_true = rng.choice(3, size=100)
        prob = rng.dirichlet(np.ones(3), size=100)
        for n_bins in [5, 10, 15, 50]:
            ece = ece_score(y_true, prob, n_bins=n_bins)
            assert 0.0 <= ece <= 1.0

    def test_binary_case(self):
        """ECE should work for binary classification."""
        y_true = np.array([0, 0, 1, 1])
        prob = np.array([[0.9, 0.1],
                         [0.8, 0.2],
                         [0.3, 0.7],
                         [0.2, 0.8]])
        ece = ece_score(y_true, prob)
        assert 0.0 <= ece <= 1.0

    def test_single_sample(self):
        """Should not crash on a single sample."""
        y_true = np.array([0])
        prob = np.array([[0.6, 0.4]])
        ece = ece_score(y_true, prob)
        assert 0.0 <= ece <= 1.0

    def test_known_value(self):
        """Manual ECE computation for a small example."""
        # 4 samples, 2 classes, 2 bins ([0, 0.5), [0.5, 1.0])
        y_true = np.array([0, 1, 0, 1])
        prob = np.array([[0.6, 0.4],   # conf=0.6, pred=0, correct
                         [0.3, 0.7],   # conf=0.7, pred=1, correct
                         [0.8, 0.2],   # conf=0.8, pred=0, correct
                         [0.9, 0.1]])  # conf=0.9, pred=0, wrong
        # All in high-confidence bin. acc=3/4=0.75, avg_conf=mean(0.6,0.7,0.8,0.9)=0.75
        # ECE = |0.75 - 0.75| * 1.0 = 0.0
        ece = ece_score(y_true, prob, n_bins=2)
        assert ece == pytest.approx(0.0, abs=0.01)
