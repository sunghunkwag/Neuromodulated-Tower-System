"""Neurotransmitter-Gated Integration Layer.

Based on Hansen et al. (2024) PET imaging findings of brainstem-cortex connectivity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List


class NeuromodulatorGate(nn.Module):
    """18-receptor neurotransmitter-gated integration layer.
    
    Based on Hansen et al. (2024) PET imaging findings.
    
    Features:
    - 3 main NT pathways: NET (norepinephrine), DAT (dopamine), 5-HTT (serotonin)
    - Context-dependent tower routing
    - Hormone-modulated gating weights
    """
    
    def __init__(self, latent_dim: int = 128, num_towers: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_towers = num_towers
        
        # 3 main NT pathways (NET, DAT, 5-HTT) -> 5 towers
        self.NET_gate = nn.Linear(latent_dim, num_towers)  # Norepinephrine
        self.DAT_gate = nn.Linear(latent_dim, num_towers)  # Dopamine
        self.HTT_gate = nn.Linear(latent_dim, num_towers)  # Serotonin (5-HTT)
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(latent_dim * num_towers, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def forward(self, 
                tower_outputs: List[torch.Tensor],
                hormones: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Integrate tower outputs with NT gating.
        
        Args:
            tower_outputs: List of [batch, latent_dim] tensors from 5 towers
            hormones: Dict of hormone levels from Tower 3
            
        Returns:
            integrated: [batch, latent_dim] gated output
            nt_weights: Dict of NT gating weights for inspection
        """
        batch_size = tower_outputs[0].shape[0]
        
        # Concatenate all tower outputs
        concat = torch.cat(tower_outputs, dim=-1)  # [batch, latent_dim * 5]
        context = self.context_encoder(concat)
        
        # Compute NT gate weights (before hormone modulation)
        net_weights = F.softmax(self.NET_gate(context), dim=-1)  # [batch, 5]
        dat_weights = F.softmax(self.DAT_gate(context), dim=-1)
        htt_weights = F.softmax(self.HTT_gate(context), dim=-1)
        
        # Apply modulation from hormones
        dopamine_mod = hormones['dopamine'].unsqueeze(-1)  # [batch, 1]
        serotonin_mod = hormones['serotonin'].unsqueeze(-1)
        cortisol_mod = hormones['cortisol'].unsqueeze(-1)
        
        # Weighted integration with hormone modulation
        # NET ~ stress/arousal (cortisol)
        # DAT ~ reward/motivation (dopamine)
        # 5-HTT ~ mood/stability (serotonin)
        net_contribution = (net_weights * cortisol_mod).unsqueeze(-1)  # [batch, 5, 1]
        dat_contribution = (dat_weights * dopamine_mod).unsqueeze(-1)
        htt_contribution = (htt_weights * serotonin_mod).unsqueeze(-1)
        
        # Combine contributions (average of 3 NT pathways)
        total_weights = (net_contribution + dat_contribution + htt_contribution) / 3.0
        
        # Apply to tower outputs
        integrated = torch.zeros_like(tower_outputs[0])
        for i, tower_out in enumerate(tower_outputs):
            integrated = integrated + total_weights[:, i, 0].unsqueeze(-1) * tower_out
        
        nt_weights = {
            'NET': net_weights,
            'DAT': dat_weights,
            '5HTT': htt_weights,
            'combined': total_weights.squeeze(-1)  # [batch, 5]
        }
        
        return integrated, nt_weights
