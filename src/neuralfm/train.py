"""Train the masked spike transformer. Usage: python -m neuralfm.train --run-name base"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_mc_maze, synthetic
from .metrics import co_bps
from .model import MaskedSpikeTransformer, ModelConfig, masked_poisson_loss, random_time_mask


@torch.no_grad()
def predict(model, x_in: np.ndarray, device, bs=128) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(x_in), bs):
        xb = torch.from_numpy(x_in[i : i + bs]).to(device)
        outs.append(torch.exp(model(xb).float()).cpu().numpy())
    return np.concatenate(outs)


def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    d = synthetic() if args.synthetic else load_mc_maze(args.datapath, args.cache)
    n_in, n_out, T = d["train_in"].shape[2], d["train_out"].shape[2], d["train_in"].shape[1]

    # Internal dev fold (from train) drives early stopping. The official val split is
    # scored exactly once, after training, for the report.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(d["train_in"]))
    n_dev = max(1, int(0.15 * len(perm)))
    dev_i, fit_i = perm[:n_dev], perm[n_dev:]
    X, Y = d["train_in"][fit_i], d["train_out"][fit_i]
    Xd, Yd = d["train_in"][dev_i], d["train_out"][dev_i]

    cfg = ModelConfig(n_in, n_out, T, args.d_model, args.heads, args.layers, args.dropout,
                      args.pos_emb)
    model = MaskedSpikeTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(X) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, total_steps=args.epochs * steps_per_epoch
    )
    Xt, Yt = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{Path(args.out).resolve().as_posix()}/mlflow.db")
        mlflow.set_experiment("neuralfm")
        mlflow.start_run(run_name=f"{args.run_name}_s{args.seed}")
        mlflow.log_params({**cfg.to_dict(), "lr": args.lr, "mask_ratio": args.mask_ratio,
                           "mask_span": args.mask_span, "epochs": args.epochs, "seed": args.seed})
    best, best_state, bad = -1e9, None, 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(Xt), device=device)
        for s in range(steps_per_epoch):
            idx = order[s * args.bs : (s + 1) * args.bs]
            xb, yb = Xt[idx], Yt[idx]
            mask = random_time_mask(len(idx), T, args.mask_ratio, args.mask_span, device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                lr_ = model(xb, mask)
            loss, _, _ = masked_poisson_loss(lr_.float(), xb, yb, mask, n_in)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        if ep % 5 == 4 or ep == args.epochs - 1:
            score = co_bps(predict(model, Xd, device)[..., n_in:], Yd, Y)
            if use_mlflow:
                mlflow.log_metrics({"loss": loss.item(), "dev_co_bps": score}, step=ep)
            print(f"ep {ep+1:3d} loss {loss.item():.4f} dev co-bps {score:.4f} "
                  f"({time.time()-t0:.0f}s)")
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
    model.load_state_dict(best_state)
    rates = predict(model, d["val_in"], device)
    # NB: "val_co_bps" here uses the stricter training-mean null. The NLB-official score (null =
    # mean of the evaluated spikes) is re-computed from the checkpoint in ablate.py/notebooks.
    res = {"run": args.run_name, "seed": args.seed, "dev_co_bps": best,
           "val_co_bps": co_bps(rates[..., n_in:], d["val_out"], d["train_out"]),
           "n_params": sum(p.numel() for p in model.parameters()),
           "train_seconds": time.time() - t0}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.run_name}_s{args.seed}.json").write_text(json.dumps(res, indent=2))
    torch.save({"cfg": cfg.to_dict(), "state": model.state_dict()},
               out / f"{args.run_name}_s{args.seed}.pt")
    if use_mlflow:
        mlflow.log_metrics({"val_co_bps": res["val_co_bps"]})
        mlflow.end_run()
    print(json.dumps(res, indent=2))
    return res


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="base")
    p.add_argument("--datapath",
                   default=os.environ.get("NEURALFM_DATA", "data/000128/sub-Jenkins"),
                   help="folder with the MC_Maze NWB files (DANDI 000128); env NEURALFM_DATA")
    p.add_argument("--cache", default="results/mc_maze_5ms.npz")
    p.add_argument("--out", default="results")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=8,
                   help="early-stopping patience in dev evaluations (one every 5 epochs)")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--pos-emb", default="learned")
    p.add_argument("--mask-ratio", type=float, default=0.25)
    p.add_argument("--mask-span", type=int, default=5)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
