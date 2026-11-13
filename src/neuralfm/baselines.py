"""Non-neural baselines for co-smoothing."""
import numpy as np
from scipy.signal.windows import gaussian
from sklearn.linear_model import PoissonRegressor


def smooth(x: np.ndarray, sd_bins: float) -> np.ndarray:
    w = gaussian(int(sd_bins * 6) | 1, sd_bins)
    w /= w.sum()
    return np.apply_along_axis(lambda v: np.convolve(v, w, "same"), 1, x)


def mean_rate(d: dict) -> np.ndarray:
    """Null model: training-set mean rate per held-out neuron."""
    n, t = d["val_in"].shape[:2]
    return np.broadcast_to(d["train_out"].mean(axis=(0, 1)), (n, t, d["train_out"].shape[2])).copy()


def smooth_poisson(d: dict, sd_ms=50, bin_ms=5, alpha=0.01) -> np.ndarray:
    """Gaussian-smooth held-in, per-neuron Poisson GLM to held-out.
    Defaults are the NLB reference settings for mc_maze (50 ms kernel, alpha 0.01)."""
    sd = sd_ms / bin_ms
    n_out = d["train_out"].shape[2]
    tr = np.log(smooth(d["train_in"], sd).reshape(-1, d["train_in"].shape[2]) + 1e-4)
    va = smooth(d["val_in"], sd)
    va_flat = np.log(va.reshape(-1, va.shape[2]) + 1e-4)
    ytr = d["train_out"].reshape(-1, n_out)
    out = np.zeros((*va.shape[:2], n_out))
    for c in range(n_out):
        m = PoissonRegressor(alpha=alpha, max_iter=500).fit(tr, ytr[:, c])
        out[:, :, c] = m.predict(va_flat).reshape(va.shape[:2])
    return out
