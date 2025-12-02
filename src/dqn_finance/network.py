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
    elif name == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {name}")

class QNetwork(nn.Module):
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

        # ... (Validation checks remain the same) ...
        if lookback * num_features != input_dim:
             raise ValueError(f"Input dim mismatch")

        self.lookback = lookback
        self.num_features = num_features
        self.d_model = d_model

        # 1. Input Projection
        self.input_projection = nn.Linear(num_features, d_model)
        
        # 2. Positional Embedding
        self.pos_embedding = nn.Embedding(lookback, d_model)
        
        # 3. Layer Normalization (CRITICAL for Transformers)
        # Normalizes the embedding before it enters the encoder
        self.layernorm_embedding = nn.LayerNorm(d_model)

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False, 
            norm_first=True # Usually stabilizes training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 5. MLP Head construction (FIXED)
        head_layers: List[nn.Module] = []
        in_dim = d_model
        
        # Iterate through all layers
        for i, (out_dim, act_name) in enumerate(zip(layer_sizes, activations)):
            head_layers.append(nn.Linear(in_dim, out_dim))
            
            # CHECK: Is this the last layer?
            is_last_layer = (i == len(layer_sizes) - 1)
            
            if not is_last_layer:
                # Only apply activation to hidden layers
                # FIX: Use the variable act_name, not hardcoded 'relu'
                act_func = activation_from_name(act_name)
                head_layers.append(act_func)
                # Optional: Add Dropout in MLP head
                head_layers.append(nn.Dropout(dropout))
            
            in_dim = out_dim

        self.head = nn.Sequential(*head_layers)
        
        # Initialize weights (Good practice for Transformers)
        self._init_weights()

    def _init_weights(self):
        """Kaiming init for Linear, restricted range for Embeddings."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Reshape: (B, L*F) -> (B, L, F)
        x = x.view(batch_size, self.lookback, self.num_features)
        
        # Project: (B, L, F) -> (B, L, d_model)
        x = self.input_projection(x)
        
        # Add Positional Embeddings
        positions = torch.arange(self.lookback, device=x.device).unsqueeze(0) # (1, L)
        pos_emb = self.pos_embedding(positions) # (1, L, d_model)
        
        x = x + pos_emb
        
        # Apply Norm before Transformer (Stabilizes gradients)
        x = self.layernorm_embedding(x)

        # Transpose for Transformer: (B, L, E) -> (L, B, E)
        x = x.transpose(0, 1)

        # Encode
        # (L, B, E)
        encoded = self.encoder(x)

        # Take last token
        # (B, E)
        last_token = encoded[-1]

        # MLP Head
        q_values = self.head(last_token)

        return q_values