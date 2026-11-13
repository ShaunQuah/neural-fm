"""Ablation sweep around the frozen step-5 config.
Resumable: finished (config, seed) pairs are skipped.

Usage: python -m neuralfm.ablate --seeds 0 1 2
Each run trains on the same internal dev-fold protocol as train.py; the official validation split
is scored once per run (NLB-official co-bps) from the saved best-dev checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_mc_maze
from .metrics import co_bps
from .model import MaskedSpikeTransformer, ModelConfig
from .train import parser, predict, run

BASE = ["--dropout", "0.5", "--wd", "0.1", "--mask-ratio", "0.25", "--mask-span", "5",
        "--epochs", "400", "--patience", "100"]

# name -> (group, overrides of BASE). "base" is reused from results/step5 ("final" runs).
CONFIGS = {
    "base": ("reference", []),
    "no_mask": ("masking", ["--mask-ratio", "0"]),
    "mask_0.10": ("masking", ["--mask-ratio", "0.10"]),
    "mask_0.50": ("masking", ["--mask-ratio", "0.50"]),
    "span_1": ("masking", ["--mask-span", "1"]),
    "span_10": ("masking", ["--mask-span", "10"]),
    "no_pos_emb": ("position", ["--pos-emb", "none"]),
    "heads_1": ("heads", ["--heads", "1"]),
    "heads_2": ("heads", ["--heads", "2"]),
    "heads_8": ("heads", ["--heads", "8"]),
    "layers_1": ("depth", ["--layers", "1"]),
    "layers_2": ("depth", ["--layers", "2"]),
    "layers_6": ("depth", ["--layers", "6"]),
    "width_64": ("width", ["--d-model", "64"]),
    "width_256": ("width", ["--d-model", "256"]),
}


def official_val(ckpt: Path, d: dict, device: str) -> float:
    blob = torch.load(ckpt, map_location=device)
    cfg = ModelConfig(**blob["cfg"])
    model = MaskedSpikeTransformer(cfg).to(device)
    model.load_state_dict(blob["state"])
    rates = predict(model, d["val_in"], device)
    return float(co_bps(rates[..., cfg.n_in:], d["val_out"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="results/ablations")
    ap.add_argument("--base-dir", default="results/step5")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = load_mc_maze(parser().get_default("datapath"), parser().get_default("cache"))

    rows = []
    for name, (group, over) in CONFIGS.items():
        if a.only and name not in a.only:
            continue
        for seed in a.seeds:
            if name == "base":
                src, run_name = Path(a.base_dir), "final"
            else:
                src, run_name = out, name
            js, ck = src / f"{run_name}_s{seed}.json", src / f"{run_name}_s{seed}.pt"
            if not js.exists():
                args = parser().parse_args(
                    ["--run-name", run_name, "--seed", str(seed), "--out", str(src)] + BASE + over)
                run(args)
            res = json.loads(js.read_text())
            if "val_co_bps_official" not in res:
                res["val_co_bps_official"] = official_val(ck, d, device)
                js.write_text(json.dumps(res, indent=2))
            rows.append({"config": name, "group": group, "seed": seed, **res})
            print(f"{name:10s} s{seed}  dev {res['dev_co_bps']:.4f}  "
                  f"val(official) {res['val_co_bps_official']:.4f}", flush=True)

    summary = {}
    for name in dict.fromkeys(r["config"] for r in rows):
        rs = [r for r in rows if r["config"] == name]
        v = np.array([r["val_co_bps_official"] for r in rs])
        dv = np.array([r["dev_co_bps"] for r in rs])
        summary[name] = {"group": rs[0]["group"], "n_seeds": len(rs), "n_params": rs[0]["n_params"],
                         "val_mean": v.mean(), "val_sd": v.std(ddof=1) if len(v) > 1 else None,
                         "dev_mean": dv.mean(), "val_per_seed": v.tolist(),
                         "dev_per_seed": dv.tolist()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nconfig        params   dev     val (mean ± sd)")
    for k, s in summary.items():
        sd = f"{s['val_sd']:.4f}" if s["val_sd"] is not None else "  n/a "
        print(f"{k:12s} {s['n_params']:8d}  {s['dev_mean']:.4f}  {s['val_mean']:.4f} ± {sd}")


if __name__ == "__main__":
    main()
