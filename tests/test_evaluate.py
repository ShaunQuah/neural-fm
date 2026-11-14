import numpy as np
import torch

from neuralfm.data import synthetic
from neuralfm.evaluate import main, score
from neuralfm.model import MaskedSpikeTransformer, ModelConfig


def _save_tiny_ckpt(path):
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=1)
    m = MaskedSpikeTransformer(cfg)
    torch.save({"cfg": cfg.to_dict(), "state": m.state_dict()}, path)


def test_score_runs_and_is_finite(tmp_path):
    ckpt = tmp_path / "m.pt"
    _save_tiny_ckpt(ckpt)
    d = synthetic(n_trials=16)
    d = {"val_in": d["train_in"], "val_out": d["train_out"]}
    assert np.isfinite(score(ckpt, d))


def test_missing_data_exits_with_message(tmp_path, capsys):
    ckpt = tmp_path / "m.pt"
    _save_tiny_ckpt(ckpt)
    rc = main(["--ckpt", str(ckpt), "--cache", str(tmp_path / "nope.npz")])
    assert rc == 2 and "No data" in capsys.readouterr().err
