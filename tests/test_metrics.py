import numpy as np
import pytest

from neuralfm.metrics import co_bps, paired_bootstrap_diff


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    rates = rng.gamma(2.0, 0.05, size=(40, 30, 6))
    return rates, rng.poisson(rates).astype(float)


def test_matches_official_nlb_definition():
    nlb = pytest.importorskip("nlb_tools.evaluation")
    rates, spikes = _toy()
    assert co_bps(rates, spikes) == pytest.approx(nlb.bits_per_spike(rates, spikes), rel=1e-9)


def test_null_scores_zero_and_truth_beats_null():
    rates, spikes = _toy()
    null = np.broadcast_to(spikes.mean(axis=(0, 1)), spikes.shape)
    assert co_bps(null, spikes) == pytest.approx(0.0, abs=1e-9)
    assert co_bps(rates, spikes) > 0


def test_train_null_is_stricter_or_equal_when_train_mean_differs():
    rates, spikes = _toy()
    train = np.random.default_rng(1).poisson(0.2, size=(50, 30, 6)).astype(float)
    assert co_bps(rates, spikes, train) >= co_bps(rates, spikes) - 1e-12


def test_paired_bootstrap_identical_models_have_zero_difference():
    rates, spikes = _toy()
    mean, lo, hi = paired_bootstrap_diff(rates, rates, spikes, n_boot=20)
    assert mean == 0.0 and lo == 0.0 and hi == 0.0
