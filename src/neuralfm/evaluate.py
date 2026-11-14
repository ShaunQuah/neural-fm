"""Re-score a saved checkpoint on the official MC_Maze validation split (NLB co-bps).

Usage:
    python -m neuralfm.evaluate --ckpt models/final_s0.pt --cache results/mc_maze_5ms.npz

If a result JSON written by train.py sits next to the checkpoint (or --expected is given), the
recorded `val_co_bps_official` is printed and the run exits non-zero when the re-scored value
differs by more than --tol. The recorded numbers came from a bf16 GPU run, so a CPU re-score
differs in the 4th decimal; the default tolerance (0.002) allows for that.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .data import load_mc_maze
from .metrics import co_bps
from .model import MaskedSpikeTransformer, ModelConfig
from .train import predict


def score(ckpt: Path, d: dict, device: str = "cpu") -> float:
    blob = torch.load(ckpt, map_location=device)
    cfg = ModelConfig(**blob["cfg"])
    model = MaskedSpikeTransformer(cfg).to(device)
    model.load_state_dict(blob["state"])
    rates = predict(model, d["val_in"], device)
    return float(co_bps(rates[..., cfg.n_in :], d["val_out"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("models/final_s0.pt"))
    ap.add_argument("--cache", default="results/mc_maze_5ms.npz",
                    help="processed MC_Maze arrays (.npz); built from raw data if missing")
    ap.add_argument("--datapath", default=None, help="raw DANDI 000128 folder (only if no cache)")
    ap.add_argument("--expected", type=Path, default=None,
                    help="train.py result JSON to compare against (default: next to --ckpt)")
    ap.add_argument("--tol", type=float, default=0.002)
    a = ap.parse_args(argv)

    if not Path(a.cache).exists() and a.datapath is None:
        print(f"No data: {a.cache} not found and --datapath not given. "
              "Mount the processed cache or the raw dataset (see README, 'Docker').",
              file=sys.stderr)
        return 2
    d = load_mc_maze(a.datapath or "", a.cache)
    got = score(a.ckpt, d)
    print(f"{a.ckpt.name}: official validation co-bps = {got:.4f}  (CPU re-score)")

    exp_path = a.expected or a.ckpt.with_suffix(".json")
    if exp_path.exists():
        want = json.loads(exp_path.read_text())["val_co_bps_official"]
        ok = abs(got - want) <= a.tol
        print(f"recorded in {exp_path.name}: {want:.4f}  |diff| = {abs(got - want):.4f}  "
              f"(tol {a.tol})  -> {'MATCH' if ok else 'MISMATCH'}")
        return 0 if ok else 1
    print("no recorded result found to compare against")
    return 0


if __name__ == "__main__":
    sys.exit(main())
