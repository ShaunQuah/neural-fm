import numpy as np
from scipy.special import gammaln


def poisson_ll(rates: np.ndarray, spikes: np.ndarray) -> float:
    rates = np.clip(rates, 1e-9, None)
    return float(np.sum(spikes * np.log(rates) - rates - gammaln(spikes + 1.0)))


def co_bps(
    rates_out: np.ndarray, spikes_out: np.ndarray, train_out: np.ndarray | None = None
) -> float:
    """Co-smoothing bits/spike: improvement over a per-neuron mean-rate null, per spike.

    Arrays are (trials, time, neurons).
    train_out=None  -> NLB-official definition (nlb_tools.evaluation.bits_per_spike): the null is
                       each neuron's mean rate in the *evaluated* spikes. Use this to compare with
                       published numbers.
    train_out given -> the null is the training-set mean rate (stricter: the null never sees
                       evaluation data). Used for dev-fold early stopping in train.py.
    """
    ref = spikes_out if train_out is None else train_out
    null = np.broadcast_to(ref.mean(axis=(0, 1), keepdims=True), spikes_out.shape)
    n_spikes = spikes_out.sum()
    return (poisson_ll(rates_out, spikes_out) - poisson_ll(null, spikes_out)) / (
        n_spikes * np.log(2)
    )


def bootstrap_ci(rates_out, spikes_out, train_out=None, n_boot=200, seed=0):
    """Trial-level bootstrap 95% CI for co_bps."""
    rng = np.random.default_rng(seed)
    n = spikes_out.shape[0]
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(co_bps(rates_out[i], spikes_out[i], train_out))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_diff(rates_a, rates_b, spikes_out, n_boot=200, seed=0):
    """95% CI for co_bps(a) - co_bps(b), resampling the SAME trials for both models."""
    rng = np.random.default_rng(seed)
    n = spikes_out.shape[0]
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        diffs.append(co_bps(rates_a[i], spikes_out[i]) - co_bps(rates_b[i], spikes_out[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)
