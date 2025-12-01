# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn

from agents.helpers import SinusoidalPosEmb


class MLP(nn.Module):
    """
    Diffusion model backbone (ε-network).

    Given:
        - x     : noised action  (B, action_dim)
        - time  : timestep index (B,) or (B, 1)
        - state : state vector   (B, state_dim)

    It predicts either:
        - noise ε (if used in epsilon-prediction mode), or
        - denoised action (if used in x0-prediction mode),

    depending on how the surrounding Diffusion wrapper interprets it.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device,          # kept for API compatibility, but not used directly
        t_dim: int = 16,
    ) -> None:
        super().__init__()

        # Time embedding network
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        # Core MLP over [x, t_embed, state]
        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
        )

        self.final_layer = nn.Linear(256, action_dim)

    def forward(
        self,
        x: torch.Tensor,        # (B, action_dim)  — current noised action
        time: torch.Tensor,     # (B,) or (B, 1)   — diffusion timestep
        state: torch.Tensor,    # (B, state_dim)   — conditioning state
    ) -> torch.Tensor:
        """
        Forward pass of the ε-network.

        Returns:
            Tensor of shape (B, action_dim)
        """
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)
