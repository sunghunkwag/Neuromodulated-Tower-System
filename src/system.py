"""Neuromodulated Tower System - Main System Architecture.

Refactored to use modular tower components from src.towers subpackage.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .towers import (
    Tower1SocialMemory,
    Tower2WorkingMemory,
    Tower3Affective,
    Tower4Sensorimotor,
    Tower5MotorCoordination,
    MirrorTower
)
from .neuromodulator_gate import NeuromodulatorGate


class FiveTowerSystem(nn.Module):
    """Main 5-Tower Neuromodulated System.
    
    Architecture:
    - 5 specialized processing towers (parallel)
    - NT-gated integration layer (hormone-modulated)
    - Cortical reasoning module (action selection)
    
    Based on:
    - Hansen et al. (2024): Brainstem-cortex connectivity
    - HRM architecture: H-module + L-module dual processing
    - Dual-process theory: System 1 (towers) + System 2 (cortex)
    """
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dim: int = 128,
                 latent_dim: int = 128,
                 output_dim: int = 4,
                 device: str = 'cpu'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.device_str = device
        
        # Initialize all 5 towers
        self.tower1 = Tower1SocialMemory(input_dim, hidden_dim, latent_dim).to(device)
        self.tower2 = Tower2WorkingMemory(input_dim, hidden_dim, latent_dim).to(device)
        self.tower3 = Tower3Affective(input_dim, hidden_dim, latent_dim).to(device)
        self.tower4 = Tower4Sensorimotor(input_dim, hidden_dim, latent_dim).to(device)
        self.tower5 = Tower5MotorCoordination(input_dim, hidden_dim, latent_dim).to(device)
        
        # NT-Gating Layer
        self.nt_gate = NeuromodulatorGate(latent_dim, num_towers=5).to(device)

        # Self-reflective mirror tower for post-gating refinement
        self.mirror_tower = MirrorTower(latent_dim).to(device)

        # Normalize integrated representation before cortical planning
        self.integration_norm = nn.LayerNorm(latent_dim)

        # Hormone-informed bias to let affective state steer cortical reasoning
        self.hormone_to_context = nn.Linear(3, latent_dim)
        
        # Cortical reasoning (H-module style)
        self.cortex = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Action output [-1, 1]
        ).to(device)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process state through 5-tower system.
        
        Args:
            state: [batch, input_dim] state tensor
            
        Returns:
            action: [batch, output_dim] action output
            debug_info: Dict containing:
                - tower_outputs: List of tower latent states
                - hormones: Dict of hormone levels
                - nt_weights: Dict of NT gating weights
                - integrated: NT-gated unified representation
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

        # Modulate integrated state using hormone-informed contextual prior
        hormone_vector = torch.stack([
            hormones['dopamine'],
            hormones['serotonin'],
            hormones['cortisol']
        ], dim=-1)
        integrated = self.integration_norm(
            integrated + 0.2 * self.hormone_to_context(hormone_vector)
        )
        
        # Phase 3: Self-reflection followed by cortical reasoning
        refined, mirror_debug = self.mirror_tower(integrated)
        action = self.cortex(refined)
        
        debug_info = {
            'tower_outputs': tower_outputs,
            'hormones': hormones,
            'nt_weights': nt_weights,
            'integrated': integrated,
            'refined': refined,
            'mirror': mirror_debug
        }
        
        return action, debug_info
    
    def get_tower_names(self):
        """Get names of all towers."""
        return [
            self.tower1.name,
            self.tower2.name,
            self.tower3.name,
            self.tower4.name,
            self.tower5.name
        ]
    
    def inspect_nt_routing(self, state: torch.Tensor) -> Dict:
        """Inspect NT routing patterns for a given state."""
        with torch.no_grad():
            _, debug_info = self.forward(state)
        
        return {
            'tower_names': self.get_tower_names(),
            'nt_weights': debug_info['nt_weights'],
            'hormones': debug_info['hormones']
        }


if __name__ == '__main__':
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing FiveTowerSystem on {device}")
    
    system = FiveTowerSystem(input_dim=256, output_dim=4, device=device)
    
    # Random input
    state = torch.randn(2, 256, device=device)  # Batch size 2
    action, debug = system(state)
    
    print(f"\nAction shape: {action.shape}")
    print(f"Action range: [{action.min().item():.4f}, {action.max().item():.4f}]")
    print(f"\nHormones:")
    for name, val in debug['hormones'].items():
        print(f"  {name}: {val.mean().item():.4f}")
    print(f"\nNT routing (combined):")
    combined = debug['nt_weights']['combined']
    for i, tower_name in enumerate(system.get_tower_names()):
        print(f"  {tower_name}: {combined[0, i].item():.4f}")
    print("\n✓ Integration successful!")
