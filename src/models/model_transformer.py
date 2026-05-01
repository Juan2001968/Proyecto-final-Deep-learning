"""Transformer / Informer / Autoformer — placeholder funcional.

La variante (vanilla / informer / autoformer) se selecciona por config; aquí
se implementa la vanilla con encoder-decoder y se deja el hueco para los
mecanismos específicos (ProbSparse attention, Auto-Correlation).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import BaseForecaster


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerForecaster(BaseForecaster):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        lookback: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        variant: str = "vanilla",
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_targets=n_targets,
            lookback=lookback,
            horizon=horizon,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            variant=variant,
        )
        self.variant = variant
        self.input_proj = nn.Linear(n_features, d_model)
        self.target_proj = nn.Linear(n_targets, d_model)
        self.pos_enc = _PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.head = nn.Linear(d_model, n_targets)
        self.tgt_token = nn.Parameter(torch.zeros(1, horizon, n_targets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(informer/autoformer): cambiar self-attention por las variantes.
        src = self.pos_enc(self.input_proj(x))                          # (B, L, D)
        tgt = self.tgt_token.expand(x.size(0), -1, -1)                  # (B, H, T)
        tgt = self.pos_enc(self.target_proj(tgt))                       # (B, H, D)
        causal = nn.Transformer.generate_square_subsequent_mask(self.horizon).to(x.device)
        out = self.transformer(src, tgt, tgt_mask=causal)               # (B, H, D)
        return self.head(out)                                            # (B, H, T)
