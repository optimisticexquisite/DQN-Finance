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
    """
    Encoder-only Transformer that reads a stack of vectors (tokens),
    applies positional embeddings, and then applies an MLP to the
    **last token's** output to produce a final vector output.

    Input:  (B, T, input_dim)
    Output: (B, mlp_output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        max_seq_len: int,

        # transformer hyperparameters
        d_model: Optional[int] = None,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        activation: str = "gelu",

        # MLP head hyperparameters
        mlp_layer_sizes: Sequence[int] = (128, 64),
        mlp_activations: Sequence[str] = ("gelu", "gelu"),
        output_dim: int = None,  # final output dim
        layer_norm_eps: float = 1e-5,

        batch_first: bool = True,
    ):
        super().__init__()

        if d_model is None:
            d_model = input_dim
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if len(mlp_layer_sizes) != len(mlp_activations):
            raise ValueError("MLP sizes and activations must match")

        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.batch_first = batch_first

        # projection to model dim (if needed)
        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        # learnable positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # scale factor common in Transformers
        self.scale = math.sqrt(d_model)

        # last-token MLP head
        mlp_layers = []
        in_dim = d_model
        for hidden, act in zip(mlp_layer_sizes, mlp_activations):
            mlp_layers.append(nn.Linear(in_dim, hidden))
            activation_fn = activation_from_name(act)
            if not isinstance(activation_fn, nn.Identity):
                mlp_layers.append(activation_fn)
            in_dim = hidden

        # final projection
        if output_dim is None:
            output_dim = mlp_layer_sizes[-1]
        mlp_layers.append(nn.Linear(in_dim, output_dim))

        self.mlp_head = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        """
        Returns: (B, output_dim)
        """
        if self.batch_first:
            B, T, D = x.shape
        else:
            T, B, D = x.shape

        if T > self.max_seq_len:
            raise ValueError("Sequence too long")

        # project to transformer model dim
        x = self.input_proj(x)

        # add positional embeddings
        positions = torch.arange(T, device=x.device)
        if self.batch_first:
            pos_emb = self.pos_embedding(positions).unsqueeze(0)  # (1, T, d_model)
        else:
            pos_emb = self.pos_embedding(positions).unsqueeze(1)  # (T, 1, d_model)

        x = x * self.scale + pos_emb

        # transformer encoder forward
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # take LAST TOKEN
        last_token = x[:, -1, :] if self.batch_first else x[-1, :, :]

        # mlp head
        out = self.mlp_head(last_token)  # (B, output_dim)
        return out
