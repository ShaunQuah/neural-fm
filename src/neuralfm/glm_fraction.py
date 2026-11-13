"""Smoothing + Poisson GLM trained on the same nested labelled subsets as pretrain.py.

Regularisation (alpha) is chosen per (seed, fraction) on the same dev subset the transformer arms
use for early stopping; the official validation split is scored once with the chosen alpha.
Usage: python -m neuralfm.glm_fraction
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import PoissonRegressor

from .baselines import smooth
from .data import load_mc_maze
from .metrics import co_bps
from .pretrain import FRACS, split
from .train import parser

SD_MS, BIN_MS = 50, 5
ALPHAS = [0.01, 0.1, 1.0]


def _feat(x):
    return np.log(smooth(x, SD_MS / BIN_MS).reshape(-1, x.shape[2]) + 1e-4)


def fit_predict(x_tr, y_tr, x_list, alpha):
    ftr = _feat(x_tr)
    ytr = y_tr.reshape(-1, y_tr.shape[2])
    fs = [_feat(x) for x in x_list]
    outs = [np.zeros((len(x), x.shape[1], y_tr.shape[2])) for x in x_list]
    for c in range(y_tr.shape[2]):
        m = PoissonRegressor(alpha=alpha, max_iter=500).fit(ftr, ytr[:, c])
        for o, f, x in zip(outs, fs, x_list):
            o[:, :, c] = m.predict(f).reshape(len(x), x.shape[1])
    return outs


def main(out="results/pretrain_v2", seeds=(0, 1, 2, 3, 4)):
    out = Path(out)
    d = load_mc_maze(parser().get_default("datapath"), parser().get_default("cache"))
    rows = []
    for seed in seeds:
        fit_i, dev_i = split(d, seed)
        perm = np.random.default_rng(10_000 + seed).permutation(fit_i)
        for frac in FRACS:
            f = out / f"glm_f{frac}_s{seed}.json"
            if not f.exists():
                t0 = time.time()
                lab = perm[: max(8, round(frac * len(fit_i)))]
                dev_lab = dev_i[: max(8, round(frac * len(dev_i)))]
                x, y = d["train_in"][lab], d["train_out"][lab]
                dev_by_alpha, val_by_alpha = {}, {}
                for alpha in ALPHAS:
                    r_dev, r_val = fit_predict(x, y, [d["train_in"][dev_lab], d["val_in"]], alpha)
                    dev_by_alpha[alpha] = float(co_bps(r_dev, d["train_out"][dev_lab], y))
                    val_by_alpha[alpha] = float(co_bps(r_val, d["val_out"]))
                best = max(dev_by_alpha, key=dev_by_alpha.get)
                f.write_text(json.dumps({"mode": "glm", "frac": frac, "seed": seed, "alpha": best,
                                         "dev_co_bps": dev_by_alpha[best],
                                         "val_co_bps": val_by_alpha[best],
                                         "dev_by_alpha": dev_by_alpha}, indent=2))
                print(f"s{seed} frac {frac}: alpha {best} dev {dev_by_alpha[best]:.4f} "
                      f"val {val_by_alpha[best]:.4f} ({time.time()-t0:.0f}s)", flush=True)
            rows.append(json.loads(f.read_text()))
    (out / "glm_runs.json").write_text(json.dumps(rows, indent=2))
    for frac in FRACS:
        v = np.array([r["val_co_bps"] for r in rows if r["frac"] == frac])
        print(f"glm frac {frac}: {v.mean():.4f} ± {v.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
