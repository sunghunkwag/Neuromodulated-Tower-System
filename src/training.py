"""Training utilities for Neuromodulated Tower System."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Dict, Tuple, Optional
from .system import FiveTowerSystem


class TowerSystemLoss(nn.Module):
    """Multi-objective loss for 5-Tower system.
    
    Components:
    1. Task loss (e.g., action prediction)
    2. NT-gate entropy regularization
    3. Hormone stability regularization
    4. Tower diversity loss
    """
    
    def __init__(self, 
                 task_weight: float = 1.0,
                 entropy_weight: float = 0.1,
                 hormone_weight: float = 0.05,
                 diversity_weight: float = 0.1):
        super().__init__()
        self.task_weight = task_weight
        self.entropy_weight = entropy_weight
        self.hormone_weight = hormone_weight
        self.diversity_weight = diversity_weight
        
    def forward(self, 
                action: torch.Tensor,
                target: torch.Tensor,
                debug_info: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-objective loss.
        
        Args:
            action: [batch, action_dim] predicted actions
            target: [batch, action_dim] target actions
            debug_info: Dict with tower_outputs, hormones, nt_weights
            
        Returns:
            total_loss: Scalar loss
            loss_components: Dict of individual loss values
        """
        # 1. Task loss (MSE for continuous actions)
        task_loss = F.mse_loss(action, target)
        
        # 2. NT-gate entropy regularization (encourage diverse routing)
        nt_weights = debug_info['nt_weights']['combined']  # [batch, 5]
        # Entropy: -sum(p * log(p))
        entropy = -(nt_weights * torch.log(nt_weights + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -entropy  # Negative because we want to maximize entropy
        
        # 3. Hormone stability (prevent extreme values)
        hormones = debug_info['hormones']
        hormone_mean = (hormones['dopamine'] + hormones['serotonin'] + hormones['cortisol']) / 3.0
        hormone_loss = F.mse_loss(hormone_mean, torch.ones_like(hormone_mean) * 0.5)
        
        # 4. Tower diversity loss (encourage different tower outputs)
        tower_outputs = debug_info['tower_outputs']  # List of [batch, latent_dim]
        diversity_loss = 0.0
        for i in range(len(tower_outputs) - 1):
            for j in range(i + 1, len(tower_outputs)):
                # Cosine similarity (higher = more similar, we want to minimize)
                cos_sim = F.cosine_similarity(tower_outputs[i], tower_outputs[j], dim=-1).mean()
                diversity_loss += cos_sim
        diversity_loss /= 10.0  # Normalize (5 choose 2 = 10 pairs)
        
        # Total loss
        total_loss = (
            self.task_weight * task_loss +
            self.entropy_weight * entropy_loss +
            self.hormone_weight * hormone_loss +
            self.diversity_weight * diversity_loss
        )
        
        loss_components = {
            'task': task_loss.item(),
            'entropy': entropy_loss.item(),
            'hormone': hormone_loss.item(),
            'diversity': diversity_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_components


def train_epoch(system: FiveTowerSystem,
                optimizer: Optimizer,
                train_loader,
                loss_fn: Optional[TowerSystemLoss] = None,
                device: str = 'cpu') -> Dict[str, float]:
    """Train for one epoch.
    
    Args:
        system: FiveTowerSystem instance
        optimizer: PyTorch optimizer
        train_loader: DataLoader with (state, target) batches
        loss_fn: Loss function (default: TowerSystemLoss)
        device: 'cpu' or 'cuda'
        
    Returns:
        epoch_metrics: Dict of averaged metrics
    """
    if loss_fn is None:
        loss_fn = TowerSystemLoss()
    
    system.train()
    epoch_losses = {'task': 0.0, 'entropy': 0.0, 'hormone': 0.0, 'diversity': 0.0, 'total': 0.0}
    num_batches = 0
    
    for state, target in train_loader:
        state = state.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        action, debug_info = system(state)
        
        # Compute loss
        loss, loss_components = loss_fn(action, target, debug_info)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        for key in epoch_losses:
            epoch_losses[key] += loss_components[key]
        num_batches += 1
    
    # Average over batches
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def evaluate(system: FiveTowerSystem,
             val_loader,
             loss_fn: Optional[TowerSystemLoss] = None,
             device: str = 'cpu') -> Dict[str, float]:
    """Evaluate on validation set.
    
    Args:
        system: FiveTowerSystem instance
        val_loader: DataLoader with (state, target) batches
        loss_fn: Loss function
        device: 'cpu' or 'cuda'
        
    Returns:
        val_metrics: Dict of validation metrics
    """
    if loss_fn is None:
        loss_fn = TowerSystemLoss()
    
    system.eval()
    val_losses = {'task': 0.0, 'entropy': 0.0, 'hormone': 0.0, 'diversity': 0.0, 'total': 0.0}
    num_batches = 0
    
    with torch.no_grad():
        for state, target in val_loader:
            state = state.to(device)
            target = target.to(device)
            
            action, debug_info = system(state)
            loss, loss_components = loss_fn(action, target, debug_info)
            
            for key in val_losses:
                val_losses[key] += loss_components[key]
            num_batches += 1
    
    for key in val_losses:
        val_losses[key] /= num_batches
    
    return val_losses
