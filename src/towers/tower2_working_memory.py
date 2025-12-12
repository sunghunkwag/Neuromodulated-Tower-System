"""Tower 2: Working Memory and Cognitive Control."""

import torch
import torch.nn as nn
from .tower_base import TowerBase


class Tower2WorkingMemory(TowerBase):
    """Tower 2: Working memory and cognitive control.
    
    Features:
    - Meta-learning capability
    - Task switching and cognitive flexibility
    - Dynamic attention mechanisms
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 128):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.name = "Working-Memory"
        
        # Meta-learning capability
        self.meta_learner = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Attention mechanism for working memory
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = super().forward(x)
        
        # Meta-learning attention
        meta_signal = torch.sigmoid(self.meta_learner(latent))
        modulated = latent * meta_signal
        
        # Self-attention for working memory coherence
        if len(modulated.shape) == 2:
            modulated = modulated.unsqueeze(1)  # [batch, 1, latent]
        
        attended, _ = self.attention(modulated, modulated, modulated)
        return attended.squeeze(1) if attended.shape[1] == 1 else attended
