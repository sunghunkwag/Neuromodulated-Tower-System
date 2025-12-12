"""Tower 4: Sensorimotor Integration."""

import torch
import torch.nn as nn
from .tower_base import TowerBase


class Tower4Sensorimotor(TowerBase):
    """Tower 4: Sensorimotor integration.
    
    Features:
    - Visual perception processing
    - Proprioceptive integration
    - Dual-head architecture (perception + action binding)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 128):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.name = "Sensorimotor"
        
        # Dual-head for perception processing
        self.perception_head = nn.Linear(latent_dim, latent_dim)
        self.action_binding = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = super().forward(x)
        
        # Perception pathway
        perception = torch.relu(self.perception_head(latent))
        
        # Action binding pathway
        action_bind = torch.tanh(self.action_binding(latent))
        
        # Combine both pathways
        return (perception + action_bind) / 2.0
