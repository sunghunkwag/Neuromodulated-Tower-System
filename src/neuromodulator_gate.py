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
    
    def __init__(
        self,
        latent_dim: int = 128,
        num_towers: int = 5,
        temperature: float = 1.0,
        min_pathway_share: float = 0.02,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_towers = num_towers
        self.temperature = temperature
        self.min_pathway_share = min_pathway_share

        # 3 main NT pathways (NET, DAT, 5-HTT) -> 5 towers
        self.NET_gate = nn.Linear(latent_dim, num_towers)  # Norepinephrine
        self.DAT_gate = nn.Linear(latent_dim, num_towers)  # Dopamine
        self.HTT_gate = nn.Linear(latent_dim, num_towers)  # Serotonin (5-HTT)

        # Learnable receptor sensitivities (pathway x tower)
        self.receptor_sensitivity = nn.Parameter(torch.randn(3, num_towers))

        # Context encoder that builds a fused view of tower outputs
        self.context_encoder = nn.Sequential(
            nn.Linear(latent_dim * num_towers, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.context_norm = nn.LayerNorm(latent_dim)

        # Baseline routing logits used to keep routing well-behaved under noisy hormones
        self.baseline_router = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_towers)
        )
        
    def forward(self,
                tower_outputs: List[torch.Tensor],
                hormones: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Integrate tower outputs with NT gating.
        
        Args:
            tower_outputs: List of [batch, latent_dim] tensors from 5 towers
            hormones: Dict of hormone levels from Tower 3
            
        Returns:
            integrated: [batch, latent_dim] gated output
            nt_weights: Dict of NT gating weights for inspection
            pathway_strength: [batch, 3] aggregate NT contributions per pathway
        """
        # Concatenate all tower outputs -> contextual summary
        concat = torch.cat(tower_outputs, dim=-1)  # [batch, latent_dim * 5]
        context = self.context_norm(self.context_encoder(concat))

        # Pathway-specific logits (before hormone scaling)
        temperature = max(1e-4, float(self.temperature))
        net_logits = self.NET_gate(context) / temperature
        dat_logits = self.DAT_gate(context) / temperature
        htt_logits = self.HTT_gate(context) / temperature

        # Convert to normalized pathway weights
        net_weights = F.softmax(net_logits, dim=-1)  # [batch, num_towers]
        dat_weights = F.softmax(dat_logits, dim=-1)
        htt_weights = F.softmax(htt_logits, dim=-1)
        pathway_weights = torch.stack([net_weights, dat_weights, htt_weights], dim=1)

        # Build hormone tensor [batch, 3, 1] aligned with pathway dimension
        hormone_levels = torch.stack([
            hormones['cortisol'],  # aligns with NET
            hormones['dopamine'],  # aligns with DAT
            hormones['serotonin']  # aligns with 5-HTT
        ], dim=1).unsqueeze(-1)

        # Learnable receptor sensitivities (broadcast to batch)
        receptor_profile = torch.sigmoid(self.receptor_sensitivity).unsqueeze(0)

        # Combine pathway routing, receptor sensitivity, and hormone levels
        modulated_weights = pathway_weights * hormone_levels * receptor_profile

        # Add a baseline router to keep routing stable when hormones are flat
        baseline_logits = self.baseline_router(context)
        baseline_weights = F.softmax(baseline_logits, dim=-1).unsqueeze(1)

        # Aggregate and normalize for safe integration
        total_weights = (modulated_weights + 0.25 * baseline_weights).sum(dim=1)

        # Prevent any tower from being fully zeroed out to improve gradient flow
        total_weights = total_weights + self.min_pathway_share
        total_weights = total_weights / (total_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Stack tower outputs for efficient weighted sum: [batch, towers, latent]
        stacked_outputs = torch.stack(tower_outputs, dim=1)
        integrated = torch.einsum('bti,bt->bi', stacked_outputs, total_weights)

        nt_weights = {
            'NET': net_weights,
            'DAT': dat_weights,
            '5HTT': htt_weights,
            'combined': total_weights,  # [batch, num_towers]
        }

        pathway_strength = modulated_weights.sum(dim=-1)  # [batch, 3]

        return integrated, nt_weights, pathway_strength
