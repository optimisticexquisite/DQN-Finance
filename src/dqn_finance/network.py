"""Neural network components for DQN trading agents."""

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

import math
from typing import Optional

import torch
from torch import nn
_ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "linear": nn.Identity,
}


class QNetwork(nn.Module):
    """Encoder-only Transformer that maps a stack of vectors to a stack of vectors.

    Expected input shape: (batch_size, seq_len, input_dim)
    Output shape:        (batch_size, seq_len, input_dim)
    """

    def __init__(
        self,
        input_dim: int,
        max_seq_len: int,
        d_model: Optional[int] = None,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",  # 'relu' or 'gelu' etc. (PyTorch-supported)
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        if d_model is None:
            d_model = input_dim

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )

        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1")

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be >= 1")

        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.batch_first = batch_first

        # Project input_dim -> d_model if needed (still "raw" vectors, no token embedding lookup)
        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
            self.output_proj = nn.Linear(d_model, input_dim)
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()

        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Core Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,  # MLP hidden size
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Optional: scale to stabilize (common for Transformers)
        self.scale = math.sqrt(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim) if batch_first=True
               or (seq_len, batch_size, input_dim) if batch_first=False.
            src_key_padding_mask: Optional bool mask of shape
                (batch_size, seq_len) indicating padding tokens to ignore.

        Returns:
            Tensor of same shape as x.
        """
        if self.batch_first:
            batch_size, seq_len, in_dim = x.shape
        else:
            seq_len, batch_size, in_dim = x.shape

        if in_dim != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {in_dim}"
            )

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        # Project to model dimension
        x = self.input_proj(x)

        # Add positional embeddings
        # positions: (seq_len,) -> broadcast appropriately
        positions = torch.arange(
            seq_len, device=x.device, dtype=torch.long
        )  # [0, 1, ..., seq_len-1]

        if self.batch_first:
            # pos_emb: (1, seq_len, d_model) broadcast across batch
            pos_emb = self.pos_embedding(positions).unsqueeze(0)
        else:
            # pos_emb: (seq_len, 1, d_model) broadcast across batch
            pos_emb = self.pos_embedding(positions).unsqueeze(1)

        x = x * self.scale + pos_emb

        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Project back to input_dim (if necessary)
        x = self.output_proj(x)

        return x