"""Masked spike transformer.

Tokenisation: each 5 ms time bin is one token. The counts of all held-in neurons in that bin
are linearly projected to d_model. Masked bins are replaced by a learned mask token before
the encoder sees them. The decoder predicts log firing rates for held-in AND held-out neurons.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    n_in: int = 137
    n_out: int = 45
    seq_len: int = 140
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.2
    pos_emb: str = "learned"  # "learned" | "none" (used in ablations)

    def to_dict(self) -> dict:
        return asdict(self)


class MaskedSpikeTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Linear(cfg.n_in, d)
        self.mask_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pos = nn.Parameter(torch.zeros(cfg.seq_len, d)) if cfg.pos_emb == "learned" else None
        if self.pos is not None:
            nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, 4 * d, cfg.dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.n_in + cfg.n_out)

    def forward(self, spikes: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """spikes: (B, T, n_in) counts. mask: (B, T) bool, True = hide this bin.
        Returns log-rates (B, T, n_in + n_out)."""
        x = self.embed(torch.log1p(spikes))
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(x), x)
        if self.pos is not None:
            x = x + self.pos[: x.shape[1]]
        return self.head(self.norm(self.encoder(x)))


def random_time_mask(batch: int, seq_len: int, ratio: float, span: int, device) -> torch.Tensor:
    """Contiguous-span masking: each span start is Bernoulli(ratio/span).

    `ratio` is nominal. Spans overlap and are cut off at the end of the trial, so the realised
    masked fraction is lower: about 0.095 / 0.22 / 0.40 for ratio 0.10 / 0.25 / 0.50 at span 5.
    """
    starts = torch.rand(batch, seq_len, device=device) < (ratio / span)
    mask = starts.clone()
    for k in range(1, span):
        mask[:, k:] |= starts[:, : seq_len - k]
    return mask


def masked_poisson_loss(log_rate, target_in, target_out, mask, n_in, out_weight=1.0):
    """Poisson NLL. Held-in: only at masked bins (otherwise it is an identity task).
    Held-out neurons are never an input, so every bin counts."""
    nll = nn.functional.poisson_nll_loss
    lr_in, lr_out = log_rate[..., :n_in], log_rate[..., n_in:]
    m = mask.unsqueeze(-1).expand_as(target_in)
    if m.any():
        loss_in = nll(lr_in[m], target_in[m], log_input=True, full=False, reduction="mean")
    else:  # no-masking ablation: held-in reconstruction is trivial, so only held-out is scored
        loss_in = lr_in.sum() * 0.0
    loss_out = nll(lr_out, target_out, log_input=True, full=False, reduction="mean")
    return loss_in + out_weight * loss_out, loss_in.detach(), loss_out.detach()
