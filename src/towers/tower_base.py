"""Base class for all processing towers."""

import torch
import torch.nn as nn


class TowerBase(nn.Module):
    """Base class for all processing towers."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Lightweight deep encoder with normalization for stability
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        hidden = self.norm(hidden)
        return self.latent(hidden)
