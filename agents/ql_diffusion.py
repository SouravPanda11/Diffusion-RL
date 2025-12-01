# ql_diffusion.py (refined)
# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA


# ───────────────────────── Critic Network ───────────────────────── #

class Critic(nn.Module):
    """
    Twin Q-network critic:
        Q1(s, a), Q2(s, a) with Mish activations.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        input_dim = state_dim + action_dim

        def make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1_model = make_head()
        self.q2_model = make_head()

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Q1(s, a), Q2(s, a)  both shape (B, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


# ───────────────────────── Diffusion Q-Learning ───────────────────────── #

class Diffusion_QL:
    """
    Main algorithm from "Diffusion Policies as an Expressive Policy Class for Offline RL".

    Components:
        - Diffusion actor over actions a|s
        - Twin Q-critic + target network
        - EMA version of actor for conservative target backups

    Key ideas:
        - Actor is trained with both:
            * diffusion BC loss
            * Q-learning term encouraging high-Q actions
        - Critic is trained with standard TD targets using EMA policy.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device,
        discount: float,
        tau: float,
        max_q_backup: bool = False,
        eta: float = 1.0,
        beta_schedule: str = "linear",
        n_timesteps: int = 100,
        ema_decay: float = 0.995,
        step_start_ema: int = 1000,
        update_ema_every: int = 5,
        lr: float = 3e-4,
        lr_decay: bool = False,
        lr_maxt: int = 1000,
        grad_norm: float = 1.0,
    ) -> None:

        # ε-network backbone
        self.model = MLP(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )

        # Diffusion actor
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
            lr=lr,
        )

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        # EMA of actor (used for target Q backup)
        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        # Critic + target critic
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # Learning-rate schedulers (optional)
        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer, T_max=lr_maxt, eta_min=0.0
            )
            self.critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer, T_max=lr_maxt, eta_min=0.0
            )

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta             # weight for Q-learning term in actor loss
        self.device = device
        self.max_q_backup = max_q_backup

    # ------------------------------------------------------------------ #
    # EMA utilities
    # ------------------------------------------------------------------ #

    def step_ema(self) -> None:
        """
        Update EMA actor after a warmup period.
        """
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    # ------------------------------------------------------------------ #
    # Training loop (one call = many gradient steps)
    # ------------------------------------------------------------------ #

    def train(
        self,
        replay_buffer,
        iterations: int,
        batch_size: int = 100,
        log_writer=None,
    ) -> Dict[str, list[float]]:
        """
        Main training method; performs `iterations` gradient steps.

        replay_buffer.sample(batch_size) must return:
            state, action, next_state, reward, not_done
        """

        metric = {
            "bc_loss": [],
            "ql_loss": [],
            "actor_loss": [],
            "critic_loss": [],
        }

        for _ in range(iterations):
            # ---------------- Q Training ---------------- #
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # Current Q estimates for dataset actions
            current_q1, current_q2 = self.critic(state, action)

            # Target Q via EMA policy
            if self.max_q_backup:
                # Sample multiple actions per next_state and take max over actions
                repeats = 10
                next_state_rpt = torch.repeat_interleave(next_state, repeats=repeats, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)  # diffusion sample

                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, repeats).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, repeats).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                # Single action sample from EMA policy
                next_action = self.ema_model(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            # TD target
            target_q = (reward + not_done * self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
                current_q2, target_q
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=self.grad_norm, norm_type=2
                )
            self.critic_optimizer.step()

            # ---------------- Policy Training ---------------- #
            # Diffusion BC loss (behavior cloning in diffusion space)
            bc_loss = self.actor.loss(action, state)

            # Sample new actions from current actor
            new_action = self.actor(state)

            # Q-learning augmentation: encourage high-Q actions
            q1_new_action, q2_new_action = self.critic(state, new_action)

            # Randomly pick q1 or q2 as numerator to avoid bias
            if np.random.uniform() > 0.5:
                q_loss = -q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = -q2_new_action.mean() / q1_new_action.abs().mean().detach()

            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=self.grad_norm, norm_type=2
                )
            self.actor_optimizer.step()

            # ---------------- EMA + target critic update ---------------- #
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # Soft-update critic target
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

            self.step += 1

            # ---------------- Logging ---------------- #
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar(
                        "Actor Grad Norm", actor_grad_norms.max().item(), self.step
                    )
                    log_writer.add_scalar(
                        "Critic Grad Norm", critic_grad_norms.max().item(), self.step
                    )
                log_writer.add_scalar("BC Loss", bc_loss.item(), self.step)
                log_writer.add_scalar("QL Loss", q_loss.item(), self.step)
                log_writer.add_scalar("Critic Loss", critic_loss.item(), self.step)
                log_writer.add_scalar("Target_Q Mean", target_q.mean().item(), self.step)

            metric["actor_loss"].append(actor_loss.item())
            metric["bc_loss"].append(bc_loss.item())
            metric["ql_loss"].append(q_loss.item())
            metric["critic_loss"].append(critic_loss.item())

        # Optional cosine LR decay (one step per train call)
        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    # ------------------------------------------------------------------ #
    # Action sampling for evaluation
    # ------------------------------------------------------------------ #

    def sample_action(self, state: np.ndarray) -> np.ndarray:
        """
        Sample an action for a single state using the actor, with Q-weighted selection.

        Strategy:
            - Repeat the state 50 times
            - Sample 50 actions from the diffusion actor
            - Evaluate Q_target(s, a)
            - Sample one action from softmax over Q-values
        """
        state_t = torch.tensor(
            state.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        state_rpt = torch.repeat_interleave(state_t, repeats=50, dim=0)

        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            probs = F.softmax(q_value, dim=0)
            idx = torch.multinomial(probs, 1)

        return action[idx].cpu().numpy().flatten()

    # ------------------------------------------------------------------ #
    # Saving / loading
    # ------------------------------------------------------------------ #

    def save_model(self, directory: str, epoch: int | None = None) -> None:
        if epoch is not None:
            torch.save(self.actor.state_dict(), f"{directory}/actor_{epoch}.pth")
            torch.save(self.critic.state_dict(), f"{directory}/critic_{epoch}.pth")
        else:
            torch.save(self.actor.state_dict(), f"{directory}/actor.pth")
            torch.save(self.critic.state_dict(), f"{directory}/critic.pth")

    def load_model(self, directory: str, epoch: int | None = None) -> None:
        if epoch is not None:
            self.actor.load_state_dict(torch.load(f"{directory}/actor_{epoch}.pth"))
            self.critic.load_state_dict(torch.load(f"{directory}/critic_{epoch}.pth"))
        else:
            self.actor.load_state_dict(torch.load(f"{directory}/actor.pth"))
            self.critic.load_state_dict(torch.load(f"{directory}/critic.pth"))
