"""Neural network components for DQN trading agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


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
    try:
        activation_cls = _ACTIVATION_MAP[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unsupported activation function: {name}") from exc
    return activation_cls()


class QNetwork(nn.Module):
    """Feed-forward network for approximating the action-value function."""

    def __init__(
        self,
        input_dim: int,
        layer_sizes: Sequence[int],
        activations: Sequence[str],
    ) -> None:
        super().__init__()
        if len(layer_sizes) == 0:
            raise ValueError("layer_sizes must contain at least one layer")
        if len(layer_sizes) != len(activations):
            raise ValueError("Number of activations must match number of layers")

        layers: List[nn.Module] = []
        in_dim = input_dim
        for out_dim, activation_name in zip(layer_sizes, activations):
            layers.append(nn.Linear(in_dim, out_dim))
            activation = activation_from_name(activation_name)
            if not isinstance(activation, nn.Identity):
                layers.append(activation)
            in_dim = out_dim

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - exercised via agent tests
        return self.model(x)


