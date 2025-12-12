"""Tower 5: Motor Coordination and Behavioral Sequencing."""

import torch
import torch.nn as nn
from .tower_base import TowerBase


class Tower5MotorCoordination(TowerBase):
    """Tower 5: Motor coordination and behavioral sequencing.
    
    Features:
    - Complex behavioral sequencing
    - Temporal planning
    - Motor program execution
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 128):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.name = "Motor-Coordination"
        
        # Sequence planner
        self.sequence_planner = nn.Linear(latent_dim, latent_dim * 2)
        
        # Temporal dynamics (optional GRU for sequence modeling)
        self.temporal_gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = super().forward(x)
        
        # Generate sequence plan
        sequence = self.sequence_planner(latent)
        
        # Extract first half as current action representation
        current_action = torch.tanh(sequence[:, :self.latent_dim])
        
        return current_action
