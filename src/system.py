"""Neuromodulated Tower System - Core Architecture

Implements a 5-tower parallel processing architecture with neurotransmitter-gated
integration based on Hansen et al. (2024) brainstem-cortex connectivity findings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List


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


class Tower1SocialMemory(TowerBase):
    """Tower 1: Autobiographical memory and social cognition."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 128):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.name = "Social-Memory"
        
        # EWC (Elastic Weight Consolidation) for long-term memory
        self.register_buffer('ewc_params', None)
        self.register_buffer('ewc_fisher', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class Tower2WorkingMemory(TowerBase):
    """Tower 2: Working memory and cognitive control."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 128):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.name = "Working-Memory"
        
        # Meta-learning capability
        self.meta_learner = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = super().forward(x)
        # Meta-learning attention
        meta_signal = torch.sigmoid(self.meta_learner(latent))
        return latent * meta_signal


class Tower3Affective(nn.Module):
    """Tower 3: Affective processing with 3-hormone neuromodulation system.
    
    Models: Dopamine, Serotonin, Cortisol
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


class Tower4Sensorimotor(TowerBase):
    """Tower 4: Sensorimotor integration."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 128):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.name = "Sensorimotor"
        
        # Dual-head for perception processing
        self.perception_head = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = super().forward(x)
        return torch.relu(self.perception_head(latent))


class Tower5MotorCoordination(TowerBase):
    """Tower 5: Motor coordination and behavioral sequencing."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 128):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.name = "Motor-Coordination"
        
        # Sequence planner
        self.sequence_planner = nn.Linear(latent_dim, latent_dim * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = super().forward(x)
        sequence = self.sequence_planner(latent)
        return torch.tanh(sequence[:, :self.latent_dim])


class NeuromodulatorGate(nn.Module):
    """18-receptor neurotransmitter-gated integration layer.
    
    Based on Hansen et al. (2024) PET imaging findings.
    """
    
    def __init__(self, latent_dim: int = 128, num_towers: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_towers = num_towers
        
        # 3 main NT pathways (NET, DAT, 5-HTT) -> 5 towers
        self.NET_gate = nn.Linear(latent_dim, num_towers)  # Norepinephrine
        self.DAT_gate = nn.Linear(latent_dim, num_towers)  # Dopamine
        self.HTT_gate = nn.Linear(latent_dim, num_towers)  # Serotonin
        
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
            tower_outputs: List of [batch, latent_dim] tensors
            hormones: Dict of hormone levels from Tower 3
            
        Returns:
            integrated: [batch, latent_dim] gated output
            nt_weights: Dict of NT gating weights
        """
        batch_size = tower_outputs[0].shape[0]
        
        # Concatenate all tower outputs
        concat = torch.cat(tower_outputs, dim=-1)  # [batch, latent_dim * 5]
        context = self.context_encoder(concat)
        
        # Compute NT gate weights
        net_weights = F.softmax(self.NET_gate(context), dim=-1)
        dat_weights = F.softmax(self.DAT_gate(context), dim=-1)
        htt_weights = F.softmax(self.HTT_gate(context), dim=-1)
        
        # Apply modulation from hormones
        dopamine_mod = hormones['dopamine'].unsqueeze(-1)  # [batch, 1]
        serotonin_mod = hormones['serotonin'].unsqueeze(-1)
        cortisol_mod = hormones['cortisol'].unsqueeze(-1)
        
        # Weighted integration
        net_contribution = (net_weights * dopamine_mod).unsqueeze(-1)  # [batch, 5, 1]
        dat_contribution = (dat_weights * cortisol_mod).unsqueeze(-1)
        htt_contribution = (htt_weights * serotonin_mod).unsqueeze(-1)
        
        # Combine contributions
        total_weights = (net_contribution + dat_contribution + htt_contribution) / 3.0
        
        # Apply to tower outputs
        integrated = torch.zeros_like(tower_outputs[0])
        for i, tower_out in enumerate(tower_outputs):
            integrated = integrated + total_weights[:, i, 0].unsqueeze(-1) * tower_out
        
        nt_weights = {
            'NET': net_weights,
            'DAT': dat_weights,
            '5HTT': htt_weights
        }
        
        return integrated, nt_weights


class FiveTowerSystem(nn.Module):
    """Main 5-Tower Neuromodulated System."""
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dim: int = 128,
                 latent_dim: int = 128,
                 output_dim: int = 4,
                 device: str = 'cpu'):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device_str = device
        
        # Initialize all 5 towers
        self.tower1 = Tower1SocialMemory(input_dim, hidden_dim, latent_dim).to(device)
        self.tower2 = Tower2WorkingMemory(input_dim, hidden_dim, latent_dim).to(device)
        self.tower3 = Tower3Affective(input_dim, hidden_dim, latent_dim).to(device)
        self.tower4 = Tower4Sensorimotor(input_dim, hidden_dim, latent_dim).to(device)
        self.tower5 = Tower5MotorCoordination(input_dim, hidden_dim, latent_dim).to(device)
        
        # NT Gate
        self.nt_gate = NeuromodulatorGate(latent_dim, num_towers=5).to(device)
        
        # Cortical reasoning (H-module style)
        self.cortex = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Action output [-1, 1]
        ).to(device)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process state through 5-tower system.
        
        Args:
            state: [batch, input_dim] state tensor
            
        Returns:
            action: [batch, output_dim] action output
            debug_info: Dict with tower outputs and NT weights
        """
        # Move to device
        state = state.to(self.device_str)
        
        # Phase 1: Parallel tower processing
        t1_out = self.tower1(state)
        t2_out = self.tower2(state)
        t3_out, hormones = self.tower3(state)
        t4_out = self.tower4(state)
        t5_out = self.tower5(state)
        
        tower_outputs = [t1_out, t2_out, t3_out, t4_out, t5_out]
        
        # Phase 2: NT-Gated Integration
        integrated, nt_weights = self.nt_gate(tower_outputs, hormones)
        
        # Phase 3: Cortical Reasoning (Action selection)
        action = self.cortex(integrated)
        
        debug_info = {
            'tower_outputs': tower_outputs,
            'hormones': hormones,
            'nt_weights': nt_weights,
            'integrated': integrated
        }
        
        return action, debug_info


if __name__ == '__main__':
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    system = FiveTowerSystem(input_dim=256, output_dim=4, device=device)
    
    # Random input
    state = torch.randn(2, 256, device=device)  # Batch size 2
    action, debug = system(state)
    
    print(f"Action shape: {action.shape}")
    print(f"Action: {action}")
    print(f"Hormones: {debug['hormones']}")
    print(f"Integration successful!")
