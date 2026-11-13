"""Pretrain-then-fine-tune, framed as a label-efficiency experiment.

Stage 1 (pretrain): masked reconstruction of HELD-IN spikes only. No held-out label is used,
    and early stopping uses the held-in masked loss on the dev fold.
Stage 2 (adapt): train the held-out readout with a fraction of the labelled trials, three ways:
    scratch   random init, same recipe, same labelled trials
    finetune  init from the pretrained encoder, all weights trained
    probe     pretrained encoder frozen, only the output head trained
Early stopping in stage 2 uses a dev subset of the same fraction, so total label use scales with it.
The official validation split is scored once per run, from the best-dev checkpoint.

Usage:
    python -m neuralfm.pretrain --stage lr        # dev-only learning-rate sweep (seeds 0-2, 25%)
    python -m neuralfm.pretrain --stage lr_frac   # dev-only per-fraction lr for scratch/finetune
    python -m neuralfm.pretrain --stage main      # pretrain + adapt, resumable
Reported "val_co_bps" is the NLB-official co-bps (null = mean of the evaluated val spikes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_mc_maze
from .metrics import co_bps
from .model import MaskedSpikeTransformer, ModelConfig, masked_poisson_loss, random_time_mask
from .train import parser, predict

# frozen step-5 architecture / regularisation
CFG = dict(d_model=128, n_heads=4, n_layers=4, dropout=0.5, pos_emb="learned")
WD, MASK_RATIO, MASK_SPAN, BS = 0.1, 0.25, 5, 64
MODES = ["scratch", "finetune", "probe"]
FRACS = [0.1, 0.25, 0.5, 1.0]
LR_FRAC_GRID = {"scratch": [3e-4, 1e-3, 3e-3, 1e-2], "finetune": [3e-4, 1e-3, 3e-3]}
LR_GRID = {"scratch": [1e-3, 3e-3, 1e-2, 3e-2], "finetune": [3e-4, 1e-3, 3e-3, 1e-2],
           "probe": [3e-3, 1e-2, 3e-2, 1e-1]}


def split(d, seed):
    rng = np.random.default_rng(seed)             # same dev fold as train.py for a given seed
    perm = rng.permutation(len(d["train_in"]))
    n_dev = max(1, int(0.15 * len(perm)))
    return perm[n_dev:], perm[:n_dev]             # fit, dev


def new_model(d, device):
    n_in, n_out, T = d["train_in"].shape[2], d["train_out"].shape[2], d["train_in"].shape[1]
    cfg = ModelConfig(n_in, n_out, T, CFG["d_model"], CFG["n_heads"], CFG["n_layers"],
                      CFG["dropout"], CFG["pos_emb"])
    return MaskedSpikeTransformer(cfg).to(device), cfg


@torch.no_grad()
def heldin_masked_nll(model, Xd, device, n_in):
    """Label-free dev criterion: Poisson NLL of held-in spikes at a fixed set of masked bins."""
    model.eval()
    xb = torch.from_numpy(Xd).to(device)
    with torch.random.fork_rng(devices=[device] if device == "cuda" else []):
        torch.manual_seed(1234)
        mask = random_time_mask(len(xb), xb.shape[1], MASK_RATIO, MASK_SPAN, device)
    lr_ = model(xb, mask).float()
    zeros = torch.zeros(len(xb), xb.shape[1], lr_.shape[-1] - n_in, device=device)
    _, loss_in, _ = masked_poisson_loss(lr_, xb, zeros, mask, n_in, out_weight=0.0)
    return loss_in.item()


def pretrain(d, seed, device, out: Path, steps=8000, eval_every=200, patience=10, lr=1e-3):
    ckpt = out / f"pre_s{seed}.pt"
    if ckpt.exists():
        return ckpt
    torch.manual_seed(seed)
    fit_i, dev_i = split(d, seed)
    n_in = d["train_in"].shape[2]
    X = torch.from_numpy(d["train_in"][fit_i]).to(device)
    Xd = d["train_in"][dev_i]
    model, cfg = new_model(d, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    best, best_state, bad, order, ptr = 1e9, None, 0, torch.randperm(len(X), device=device), 0
    zeros = torch.zeros(BS, X.shape[1], cfg.n_out, device=device)
    for step in range(1, steps + 1):
        model.train()
        if ptr + BS > len(order):
            order, ptr = torch.randperm(len(X), device=device), 0
        xb, ptr = X[order[ptr:ptr + BS]], ptr + BS
        mask = random_time_mask(BS, X.shape[1], MASK_RATIO, MASK_SPAN, device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            lr_ = model(xb, mask)
        loss, _, _ = masked_poisson_loss(lr_.float(), xb, zeros, mask, n_in, out_weight=0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0:
            score = heldin_masked_nll(model, Xd, device, n_in)
            print(f"pretrain s{seed} step {step} train {loss.item():.4f} "
                  f"dev held-in NLL {score:.4f}",
                  flush=True)
            if score < best:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    torch.save({"cfg": cfg.to_dict(), "state": best_state, "dev_heldin_nll": best}, ckpt)
    return ckpt


def adapt(d, seed, mode, frac, device, lr, pre_ckpt: Path | None, steps=4000, eval_every=100,
          patience=8, trace: list | None = None):
    torch.manual_seed(seed * 1000 + int(frac * 100))
    fit_i, dev_i = split(d, seed)
    perm = np.random.default_rng(10_000 + seed).permutation(fit_i)       # nested label subsets
    lab = perm[: max(8, round(frac * len(fit_i)))]
    dev_lab = dev_i[: max(8, round(frac * len(dev_i)))]
    n_in = d["train_in"].shape[2]
    X = torch.from_numpy(d["train_in"][lab]).to(device)
    Y = torch.from_numpy(d["train_out"][lab]).to(device)
    Xd, Yd, Ylab = d["train_in"][dev_lab], d["train_out"][dev_lab], d["train_out"][lab]

    model, cfg = new_model(d, device)
    if mode in ("finetune", "probe"):
        model.load_state_dict(torch.load(pre_ckpt, map_location=device)["state"])
    if mode == "probe":
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith("head.")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    bs = min(BS, len(X))
    best, best_state, bad, order, ptr = -1e9, None, 0, torch.randperm(len(X), device=device), 0
    for step in range(1, steps + 1):
        model.train()
        if mode == "probe":
            model.eval()                               # frozen encoder: no dropout
        if ptr + bs > len(order):
            order, ptr = torch.randperm(len(X), device=device), 0
        idx, ptr = order[ptr:ptr + bs], ptr + bs
        # The probe's encoder is frozen, so there is no identity shortcut to guard against; training
        # the head on masked-input features but scoring on unmasked ones was a seed-dependent
        # distribution shift (diagnosed on the dev fold), so the probe trains unmasked.
        ratio = 0.0 if mode == "probe" else MASK_RATIO
        mask = random_time_mask(bs, X.shape[1], ratio, MASK_SPAN, device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            lr_ = model(X[idx], mask)
        loss, _, _ = masked_poisson_loss(lr_.float(), X[idx], Y[idx], mask, n_in)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0:
            score = co_bps(predict(model, Xd, device)[..., n_in:], Yd, Ylab)
            if trace is not None:
                trace.append((step, loss.item(), float(score)))
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    model.load_state_dict(best_state)
    rates = predict(model, d["val_in"], device)
    return {"mode": mode, "frac": frac, "seed": seed, "lr": lr, "n_labelled_trials": int(len(lab)),
            "n_dev_trials": int(len(dev_lab)), "dev_co_bps": float(best),
            "val_co_bps": float(co_bps(rates[..., n_in:], d["val_out"])), "steps_run": step}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["lr", "lr_frac", "main"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES,
                    help="lr stage: modes to (re)sweep; others keep their saved result")
    ap.add_argument("--out", default="results/pretrain")
    ap.add_argument("--pre-steps", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr-from", default=None,
                    help="dir holding lr_sweep.json (probe lr) and pre_s*.pt to copy")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = load_mc_maze(parser().get_default("datapath"), parser().get_default("cache"))

    if a.lr_from:                                   # reuse pretrained encoders + probe lr from v1
        import shutil
        for f in Path(a.lr_from).glob("pre_s*.pt"):
            if not (out / f.name).exists():
                shutil.copy(f, out / f.name)
        if not (out / "lr_sweep.json").exists():
            shutil.copy(Path(a.lr_from) / "lr_sweep.json", out / "lr_sweep.json")

    if a.stage == "lr_frac":
        # dev-only: per-fraction learning rate for scratch and finetune, mean dev over 3 seeds
        pres = {s: pretrain(d, s, device, out, steps=a.pre_steps) for s in (0, 1, 2)}
        lf = out / "lr_by_frac.json"
        res = json.loads(lf.read_text()) if lf.exists() else {}
        for mode in a.modes:
            if mode == "probe":
                continue
            for frac in FRACS:
                key = f"{frac}"
                if key in res.get(mode, {}):
                    continue
                dev_by_lr = {}
                for lr in LR_FRAC_GRID[mode]:
                    runs = [adapt(d, s, mode, frac, device, lr, pres[s], steps=a.steps)
                            for s in (0, 1, 2)]
                    dev_by_lr[lr] = [r["dev_co_bps"] for r in runs]
                    print(f"{mode:8s} frac {frac:<4} lr {lr:.0e}  dev mean "
                          f"{np.mean(dev_by_lr[lr]):.4f}  per-seed "
                          f"{np.round(dev_by_lr[lr], 3).tolist()}", flush=True)
                res.setdefault(mode, {})[key] = {
                    "best_lr": max(dev_by_lr, key=lambda k: np.mean(dev_by_lr[k])),
                    "dev_by_lr": {str(k): v for k, v in dev_by_lr.items()}}
                lf.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    if a.stage == "lr":
        # dev-only: pick each mode's learning rate by mean dev score over 3 seeds at 25% labels
        # (a single seed on 64 dev trials was too noisy to rank learning rates)
        pres = {s: pretrain(d, s, device, out, steps=a.pre_steps) for s in (0, 1, 2)}
        lr_file = out / "lr_sweep.json"
        res = json.loads(lr_file.read_text()) if lr_file.exists() else {}
        for mode in a.modes:
            dev_by_lr = {}
            for lr in LR_GRID[mode]:
                runs = [adapt(d, s, mode, 0.25, device, lr, pres[s], steps=a.steps)
                        for s in (0, 1, 2)]
                dev_by_lr[lr] = [r["dev_co_bps"] for r in runs]
                print(f"{mode:8s} lr {lr:.0e}  dev mean {np.mean(dev_by_lr[lr]):.4f}  "
                      f"per-seed {np.round(dev_by_lr[lr], 3).tolist()}", flush=True)
            res[mode] = {"best_lr": max(dev_by_lr, key=lambda k: np.mean(dev_by_lr[k])),
                         "dev_by_lr": {str(k): v for k, v in dev_by_lr.items()}}
        (out / "lr_sweep.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    lrs = json.loads((out / "lr_sweep.json").read_text())
    lf = out / "lr_by_frac.json"
    by_frac = json.loads(lf.read_text()) if lf.exists() else {}

    def pick_lr(mode, frac):
        return by_frac.get(mode, {}).get(f"{frac}", {}).get("best_lr", lrs[mode]["best_lr"])

    rows = []
    for seed in a.seeds:
        pre = pretrain(d, seed, device, out, steps=a.pre_steps)
        for frac in FRACS:
            for mode in MODES:
                f = out / f"{mode}_f{frac}_s{seed}.json"
                if not f.exists():
                    r = adapt(d, seed, mode, frac, device, pick_lr(mode, frac), pre, steps=a.steps)
                    f.write_text(json.dumps(r, indent=2))
                r = json.loads(f.read_text())
                rows.append(r)
                print(f"s{seed} frac {frac:<4} {mode:8s} dev {r['dev_co_bps']:.4f} "
                      f"val {r['val_co_bps']:.4f}", flush=True)
    all_f = out / "all_runs.json"      # keep rows of seeds not re-requested (partial --seeds runs)
    if all_f.exists():
        rows = [r for r in json.loads(all_f.read_text()) if r["seed"] not in a.seeds] + rows
    all_f.write_text(json.dumps(rows, indent=2))
    print("\nmode      frac   n_lab   val mean ± sd")
    for mode in MODES:
        for frac in FRACS:
            v = np.array([r["val_co_bps"] for r in rows if r["mode"] == mode and r["frac"] == frac])
            n = [r["n_labelled_trials"] for r in rows if r["mode"] == mode and r["frac"] == frac][0]
            print(f"{mode:9s} {frac:<5} {n:5d}   {v.mean():.4f} ± {v.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
