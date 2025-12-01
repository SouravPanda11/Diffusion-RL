# bc_diffusion.py (refined)
# Copyright…
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from agents.diffusion import Diffusion
from agents.model import MLP


class Diffusion_BC:
    """
    Diffusion-based Behavior Cloning:
        - No Q-learning
        - Trains only the diffusion denoiser on (state, action) pairs
        - Used as a baseline comparison model in the paper
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device,
        discount: float,
        tau: float,
        beta_schedule: str = "linear",
        n_timesteps: int = 100,
        lr: float = 2e-4,
    ) -> None:

        # ε-network backbone
        self.model = MLP(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )

        # Full diffusion model (wraps the ε-network)
        self.actor = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=self.model,
            max_action=max_action,
            beta_schedule=beta_schedule,
            n_timesteps=n_timesteps,
        ).to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr
        )

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

    # ------------------------------------------------------------------ #
    #                            Training                                 #
    # ------------------------------------------------------------------ #
    def train(
        self,
        replay_buffer,
        iterations: int,
        batch_size: int = 100,
        log_writer=None,
    ):
        """
        Trains the diffusion model purely via behavior cloning.
        No critic is used, so only the BC loss is updated.
        """

        metric = {
            "bc_loss": [],
            "ql_loss": [],
            "actor_loss": [],
            "critic_loss": [],
        }

        for _ in range(iterations):
            # Sample a batch of transitions
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # Pure BC diffusion loss
            loss = self.actor.loss(action, state)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            # Log metrics (BC only)
            metric["bc_loss"].append(loss.item())
            metric["ql_loss"].append(0.0)
            metric["actor_loss"].append(0.0)
            metric["critic_loss"].append(0.0)

        return metric

    # ------------------------------------------------------------------ #
    #                         Action Sampling                             #
    # ------------------------------------------------------------------ #
    def sample_action(self, state: np.ndarray) -> np.ndarray:
        """
        state: (state_dim,) numpy array
        returns: (action_dim,) numpy array
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.sample(state_t)
        return action.cpu().numpy().flatten()

    # ------------------------------------------------------------------ #
    #                             Saving                                   #
    # ------------------------------------------------------------------ #
    def save_model(self, directory: str, epoch: int | None = None) -> None:
        if epoch is not None:
            torch.save(self.actor.state_dict(), f"{directory}/actor_{epoch}.pth")
        else:
            torch.save(self.actor.state_dict(), f"{directory}/actor.pth")

    def load_model(self, directory: str, epoch: int | None = None) -> None:
        if epoch is not None:
            self.actor.load_state_dict(torch.load(f"{directory}/actor_{epoch}.pth"))
        else:
            self.actor.load_state_dict(torch.load(f"{directory}/actor.pth"))
