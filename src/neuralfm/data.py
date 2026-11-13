"""Data loading for the Neural Latents Benchmark (MC_Maze, DANDI 000128)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

KEYS = ["train_in", "train_out", "val_in", "val_out"]


def load_mc_maze(datapath: str, cache: str = "results/mc_maze_5ms.npz", bin_width: int = 5):
    """Return dict of (trials, time, neurons) arrays. Cached because NWB parsing is slow."""
    cache_p = Path(cache)
    if cache_p.exists():
        z = np.load(cache_p)
        return {k: z[k] for k in KEYS}
    from nlb_tools.make_tensors import make_eval_input_tensors, make_train_input_tensors
    from nlb_tools.nwb_interface import NWBDataset

    ds = NWBDataset(datapath)
    ds.resample(bin_width)
    tr = make_train_input_tensors(ds, "mc_maze", "train", save_file=False)
    va = make_eval_input_tensors(ds, "mc_maze", "val", save_file=False)
    # Held-out *val* spikes are the scoring targets. Fetched directly (no behaviour tensors needed).
    from nlb_tools.make_tensors import PARAMS, _prep_mask, make_stacked_array

    prm = PARAMS["mc_maze"]
    eval_out = make_stacked_array(ds, [prm["hospk_field"]], prm["eval_make_params"],
                                  _prep_mask(ds, "val"))
    out = {
        "train_in": tr["train_spikes_heldin"].astype(np.float32),
        "train_out": tr["train_spikes_heldout"].astype(np.float32),
        "val_in": va["eval_spikes_heldin"].astype(np.float32),
        "val_out": eval_out[prm["hospk_field"]].astype(np.float32),
    }
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_p, **out)
    return out


def synthetic(n_trials=256, T=140, n_in=137, n_out=45, seed=0):
    """Low-rank Poisson data for tests and CI (no dataset download needed)."""
    rng = np.random.default_rng(seed)
    lat = np.cumsum(rng.normal(size=(n_trials, T, 3)), axis=1) * 0.15
    w = rng.normal(size=(3, n_in + n_out)) * 0.8
    rates = np.clip(np.exp(lat @ w - 3.0), 0, 1.5)
    x = rng.poisson(rates).astype(np.float32)
    return {
        "train_in": x[: n_trials // 2, :, :n_in], "train_out": x[: n_trials // 2, :, n_in:],
        "val_in": x[n_trials // 2 :, :, :n_in], "val_out": x[n_trials // 2 :, :, n_in:],
    }
