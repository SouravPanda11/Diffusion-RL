# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Dict, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────── Time Embedding ───────────────────────── #

class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal positional embedding over scalar timesteps.

    Input:
        x : (B,) or (B, 1) tensor of timesteps

    Output:
        (B, dim) tensor of sinusoidal features.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2 and x.shape[1] == 1:
            x = x.squeeze(-1)

        device = x.device
        half_dim = self.dim // 2

        # Exponential frequencies
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)

        # Shape: (B, half_dim)
        emb = x[:, None] * freqs[None, :]
        # Concatenate sin and cos
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# ───────────────────────── Diffusion Schedules ───────────────────────── #

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Extract coefficients at indices t and reshape to broadcast over x_shape.

    a: (T,) tensor of per-timestep scalars
    t: (B,) long tensor of timestep indices
    x_shape: shape of target tensor (B, ...)

    Returns: (B, 1, 1, ..., 1) with same number of dims as x_shape.
    """
    b = t.shape[0]
    out = a.gather(-1, t)  # (B,)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Cosine schedule as proposed in:
        Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models"
        https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0.0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Simple linear beta schedule between beta_start and beta_end.
    """
    betas = np.linspace(beta_start, beta_end, timesteps)
    return torch.tensor(betas, dtype=dtype)


def vp_beta_schedule(
    timesteps: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Variance-preserving style schedule used in some diffusion works.
    """
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1.0 - alpha
    return torch.tensor(betas, dtype=dtype)


# ───────────────────────── Losses ───────────────────────── #

class WeightedLoss(nn.Module):
    """
    Base class for per-sample weighted losses.

    Expects:
        pred, targ : (B, D)
        weights    : broadcastable to (B, D) or (B,)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        targ: torch.Tensor,
        weights: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        loss = self._loss(pred, targ)          # (B, D)
        weighted_loss = (loss * weights).mean()
        return weighted_loss

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class WeightedL1(WeightedLoss):
    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, targ, reduction="none")


Losses: Dict[str, Type[WeightedLoss]] = {
    "l1": WeightedL1,
    "l2": WeightedL2,
}


# ───────────────────────── EMA (Exponential Moving Average) ───────────────────────── #

class EMA:
    """
    Exponential Moving Average over model parameters.

    Typically used to maintain a slowly-updated target / EMA model:
        ema = EMA(beta=0.995)
        ema_model = copy.deepcopy(model)
        ...
        ema.update_model_average(ema_model, model)
    """

    def __init__(self, beta: float) -> None:
        self.beta = beta

    def update_model_average(
        self,
        ma_model: nn.Module,
        current_model: nn.Module,
    ) -> None:
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight = ma_params.data
            up_weight = current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(
        self,
        old: torch.Tensor | None,
        new: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard EMA:
            old ← beta * old + (1 - beta) * new
        """
        if old is None:
            return new
        return old * self.beta + new * (1.0 - self.beta)
