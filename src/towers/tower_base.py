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
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.latent = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        return self.latent(hidden)
