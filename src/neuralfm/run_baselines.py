"""Score the non-neural baselines on the official val split. Writes results/baselines.json.

Usage: python -m neuralfm.run_baselines [--datapath data/000128/sub-Jenkins]
co-bps uses the NLB-official definition (null = mean rate of the evaluated spikes).
The null has no confidence interval: it is re-estimated inside each bootstrap resample.
"""
import argparse
import json
from pathlib import Path

from .baselines import mean_rate, smooth_poisson
from .data import load_mc_maze
from .metrics import bootstrap_ci, co_bps
from .train import parser

BASELINES = {
    "mean_rate_null": mean_rate,
    "glm_40ms_alpha0.1": lambda d: smooth_poisson(d, sd_ms=40, alpha=0.1),
    "glm_50ms_alpha0.01_nlb_reference": lambda d: smooth_poisson(d, sd_ms=50, alpha=0.01),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datapath", default=parser().get_default("datapath"))
    ap.add_argument("--cache", default=parser().get_default("cache"))
    ap.add_argument("--out", default="results/baselines.json")
    a = ap.parse_args()
    d = load_mc_maze(a.datapath, a.cache)
    out = {}
    for name, fn in BASELINES.items():
        r = fn(d)
        entry = {"val_co_bps": float(co_bps(r, d["val_out"]))}
        if name != "mean_rate_null":
            entry["ci95"] = [float(x) for x in bootstrap_ci(r, d["val_out"])]
        out[name] = entry
        print(name, entry, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
