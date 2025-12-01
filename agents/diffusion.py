# diffusion.py
# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from agents.helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
    Losses,
)
from utils.utils import Progress, Silent


class Diffusion(nn.Module):
    """
    Conditional 1D diffusion model over actions a ∈ R^{action_dim}, conditioned on state s.

    Core responsibilities:
        - Construct diffusion forward / reverse process (betas, alphas, posteriors)
        - Provide sampling:
            sample(state) → denoised action
        - Provide training loss:
            loss(action, state) → scalar

    The wrapped `model` is the ε-network / denoiser taking:
        model(x_t, t, state) → ε̂_t or x̂_0  (depending on `predict_epsilon`)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model: nn.Module,
        max_action: float,
        beta_schedule: str = "linear",   # ['linear', 'cosine', 'vp']
        n_timesteps: int = 100,
        loss_type: str = "l2",           # ['l1', 'l2']
        clip_denoised: bool = True,
        predict_epsilon: bool = True,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = float(max_action)
        self.model = model

        # -------------------- noise schedule -------------------- #
        if beta_schedule == "linear":
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == "vp":
            betas = vp_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas.dtype), alphas_cumprod[:-1]],
            dim=0,
        )

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        # Register as buffers so they move with .to(device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # q(x_t | x_0) and related terms
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod",
            torch.log(1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1.0),
        )

        # q(x_{t-1} | x_t, x_0) posterior
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        # log variance clipped at small positive value to avoid log(0)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # loss_fn: WeightedL1 or WeightedL2
        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------------------------------ #
    #                           Sampling                                  #
    # ------------------------------------------------------------------ #

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recover x_0 from x_t and predicted noise ε̂_t.

        If `predict_epsilon`:
            model output = ε̂_t  → use standard DDPM inversion formula.
        Else:
            model output is treated directly as x̂_0.
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            # In x0-prediction mode we simply treat `noise` as x̂_0
            return noise

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute q(x_{t-1} | x_t, x_0) mean and variance.
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        x: torch.Tensor,        # x_t
        t: torch.Tensor,
        s: torch.Tensor,        # state
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One reverse diffusion step: compute pθ(x_{t-1} | x_t, s).
        """
        eps_pred = self.model(x, t, s)
        x_recon = self.predict_start_from_noise(x, t=t, noise=eps_pred)

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            # In this codebase clip_denoised is always True,
            # but we keep the branch for completeness.
            raise RuntimeError("Unclipped denoising is not supported.")

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon,
            x_t=x,
            t=t,
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,        # x_t
        t: torch.Tensor,        # (B,)
        s: torch.Tensor,        # state
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from pθ(x_{t-1} | x_t, s).
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(x.shape) - 1))
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        state: torch.Tensor,          # (B, state_dim)
        shape: tuple[int, int],
        verbose: bool = False,
        return_diffusion: bool = False,
    ):
        """
        Run full reverse diffusion chain:
            x_T ~ N(0, I)
            ...
            x_0 = p_sample(...)

        Args:
            state          : conditioning states (B, state_dim)
            shape          : (B, action_dim)
            verbose        : if True, prints progress with `Progress`
            return_diffusion: if True, also returns all intermediate x_t
        """
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion:
            diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full(
                (batch_size,),
                i,
                device=device,
                dtype=torch.long,
            )
            x = self.p_sample(x, timesteps, state)

            progress.update({"t": i})

            if return_diffusion:
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample denoised actions given states.

        Args:
            state: (B, state_dim)

        Returns:
            actions: (B, action_dim), clamped to [-max_action, max_action]
        """
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-self.max_action, self.max_action)

    # ------------------------------------------------------------------ #
    #                            Training                                #
    # ------------------------------------------------------------------ #

    def q_sample(
        self,
        x_start: torch.Tensor,         # x_0 (clean actions)
        t: torch.Tensor,               # (B,)
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0).

        x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(
        self,
        x_start: torch.Tensor,     # clean actions
        state: torch.Tensor,       # states
        t: torch.Tensor,           # timesteps
        weights: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """
        Compute diffusion training loss for a batch.
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            # ε-prediction loss
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            # x0-prediction loss
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        weights: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """
        Sample random timesteps and compute Monte Carlo estimate of diffusion loss.
        """
        batch_size = x.shape[0]
        t = torch.randint(
            low=0,
            high=self.n_timesteps,
            size=(batch_size,),
            device=x.device,
            dtype=torch.long,
        )
        return self.p_losses(x, state, t, weights)

    def forward(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Alias for sample(...), so you can call:
            actions = diffusion(state)
        """
        return self.sample(state, *args, **kwargs)
