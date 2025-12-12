"""Tower 3: Affective Processing with 3-Hormone System."""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class Tower3Affective(nn.Module):
    """Tower 3: Affective processing with 3-hormone neuromodulation system.
    
    Models: Dopamine, Serotonin, Cortisol
    
    Features:
    - Emotional state representation
    - Intrinsic motivation generation
    - Hormone-based neuromodulation
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 128):
        super().__init__()
        self.name = "Affective"
        self.latent_dim = latent_dim
        
        # Emotion encoder
        self.emotion_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Hormone generation (3-dim)
        self.dopamine_head = nn.Linear(latent_dim, 1)
        self.serotonin_head = nn.Linear(latent_dim, 1)
        self.cortisol_head = nn.Linear(latent_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        latent = self.emotion_encoder(x)
        
        # Generate hormone levels [0, 1]
        dopamine = torch.sigmoid(self.dopamine_head(latent))
        serotonin = torch.sigmoid(self.serotonin_head(latent))
        cortisol = torch.sigmoid(self.cortisol_head(latent))
        
        hormones = {
            'dopamine': dopamine.squeeze(-1),
            'serotonin': serotonin.squeeze(-1),
            'cortisol': cortisol.squeeze(-1)
        }
        
        return latent, hormones
