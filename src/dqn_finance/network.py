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
def activation_from_name(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {name}")
class QNetwork(nn.Module):
    """Transformer-based network for approximating the action-value function.

    Assumes the input is a flattened sequence of vectors corresponding to a
    sliding window of OHLCV-like features:

        x.shape == (batch_size, lookback * num_features)

    Internally we reshape to:

        (batch_size, lookback, num_features)

    Each timestep vector is treated as a token (no token embedding lookup).
    We apply a linear projection to `d_model`, add positional embeddings and
    run an encoder-only transformer. The last token's hidden state is then
    fed through an MLP head to produce Q-values for each action.
    """

    def __init__(
        self,
        input_dim: int,
        layer_sizes: Sequence[int],
        activations: Sequence[str],
        *,
        lookback: int,
        num_features: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if lookback <= 0:
            raise ValueError("lookback must be positive for transformer QNetwork")
        if num_features <= 0:
            raise ValueError("num_features must be positive for transformer QNetwork")

        expected_input_dim = lookback * num_features
        if input_dim != expected_input_dim:
            raise ValueError(
                f"Transformer QNetwork expects input_dim={expected_input_dim} "
                f"(lookback={lookback} * num_features={num_features}), "
                f"but got input_dim={input_dim}."
            )

        if len(layer_sizes) == 0:
            raise ValueError("layer_sizes must contain at least one layer")
        if len(layer_sizes) != len(activations):
            raise ValueError("Number of activations must match number of layers")

        self.lookback = lookback
        self.num_features = num_features
        self.d_model = d_model

        # --- Token projection: per-timestep feature vector -> d_model ---
        # This is a simple linear projection, not an embedding lookup.
        self.input_proj = nn.Linear(num_features, d_model)

        # --- Positional embedding for sequence positions [0 .. lookback-1] ---
        self.pos_embedding = nn.Embedding(lookback, d_model)

        # --- Transformer encoder stack ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",  # or "relu" if you prefer
            batch_first=False,  # we'll feed (L, B, E)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # --- MLP head on the last token for classification/Q-values ---
        head_layers: List[nn.Module] = []
        in_dim = d_model
        for out_dim, activation_name in zip(layer_sizes, activations):
            head_layers.append(nn.Linear(in_dim, out_dim))
            activation = activation_from_name('relu')
            # Keep final activations consistent with your original design:
            if not isinstance(activation, nn.Identity):
                head_layers.append(activation)
            in_dim = out_dim

        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Tensor of shape (batch_size, lookback * num_features).

        Returns
        -------
        torch.Tensor
            Q-values for each action: shape (batch_size, n_actions),
            where n_actions == layer_sizes[-1].
        """
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D input tensor of shape (batch_size, {self.lookback * self.num_features}), "
                f"got shape {tuple(x.shape)}"
            )

        batch_size, flat_dim = x.shape
        expected_dim = self.lookback * self.num_features
        if flat_dim != expected_dim:
            raise ValueError(
                f"Expected input dimension {expected_dim}, got {flat_dim}. "
                f"Check environment lookback/feature configuration."
            )

        # Reshape flat vector to (B, L, F)
        x = x.view(batch_size, self.lookback, self.num_features)  # (B, L, F)

        # Project features to d_model
        x = self.input_proj(x)  # (B, L, d_model)

        # Add positional embeddings
        # positions: (1, L)
        positions = torch.arange(self.lookback, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)  # (1, L, d_model)
        x = x + pos_emb  # broadcast over batch -> (B, L, d_model)

        # Transformer encoder expects (L, B, d_model) since batch_first=False
        x = x.transpose(0, 1)  # (L, B, d_model)

        # Encode sequence
        encoded = self.encoder(x)  # (L, B, d_model)

        # Take last token representation: shape (B, d_model)
        last_token = encoded[-1]  # (B, d_model)

        # MLP head to get Q-values: (B, n_actions)
        q_values = self.head(last_token)
        return q_values
