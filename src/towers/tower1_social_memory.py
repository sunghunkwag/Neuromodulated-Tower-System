"""Tower 1: Social Memory and Autobiographical Processing."""

import torch
import torch.nn as nn
from .tower_base import TowerBase


class Tower1SocialMemory(TowerBase):
    """Tower 1: Autobiographical memory and social cognition.
    
    Features:
    - EWC (Elastic Weight Consolidation) for long-term memory
    - Social reasoning pathways
    - Episodic memory consolidation
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 128):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.name = "Social-Memory"
        
        # EWC (Elastic Weight Consolidation) for long-term memory
        self.register_buffer('ewc_params', None)
        self.register_buffer('ewc_fisher', None)
        
        # Social cognition head
        self.social_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = super().forward(x)
        # Apply social cognition processing
        social_signal = torch.sigmoid(self.social_head(latent))
        return latent * social_signal
    
    def consolidate_memory(self, importance_weights: torch.Tensor):
        """Store current parameters for EWC."""
        params = {n: p.clone().detach() for n, p in self.named_parameters()}
        self.ewc_params = params
        self.ewc_fisher = importance_weights
