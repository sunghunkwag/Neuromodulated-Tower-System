"""Mirror Tower: Self-reflective refinement loop for integrated latents."""

import torch
import torch.nn as nn


class MirrorTower(nn.Module):
    """Stabilize and refine integrated representations via self-reflection.

    The mirror tower maintains a running self-reflection state (exponential
    moving average of refined latents) and blends it with the current
    integrated representation. A gated residual block nudges the latent toward
    a self-consistent direction while LayerNorm keeps the dynamics stable.
    """

    def __init__(
        self,
        latent_dim: int,
        reflection_decay: float = 0.9,
        min_gate: float = 0.05,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.reflection_decay = reflection_decay
        self.min_gate = min_gate

        # Track the running self-reflection state without accumulating grads.
        self.register_buffer("self_reflection_state", torch.zeros(1, latent_dim))

        self.residual_block = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.gate = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid(),
        )

        self.stability_norm = nn.LayerNorm(latent_dim)

    def forward(self, integrated: torch.Tensor):
        """Refine the integrated latent using a self-reflection loop."""
        mirror_state = self.self_reflection_state.to(integrated.device)
        expanded_state = mirror_state.expand_as(integrated)

        # Compute gated residual update toward a self-consistent direction.
        gate_input = torch.cat([integrated, expanded_state], dim=-1)
        gate_strength = torch.clamp(self.gate(gate_input), self.min_gate, 1.0)

        candidate_update = self.residual_block(integrated)
        refined = self.stability_norm(integrated + gate_strength * (candidate_update - integrated))

        # Update running self-reflection state (EMA) without tracking gradients.
        with torch.no_grad():
            batch_mean = refined.mean(dim=0, keepdim=True)
            updated = self.reflection_decay * mirror_state + (1 - self.reflection_decay) * batch_mean
            self.self_reflection_state.copy_(updated.detach())

        mirror_debug = {
            "gate": gate_strength.detach(),
            "state": self.self_reflection_state.detach(),
            "candidate": candidate_update.detach(),
        }

        return refined, mirror_debug
