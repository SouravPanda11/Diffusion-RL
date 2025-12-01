# visualize_diffusion_policy.py
import json
import os

import numpy as np
import torch
import gymnasium as gym

from agents.bc_diffusion import Diffusion_BC
from agents.ql_diffusion import Diffusion_QL

def load_agent(result_dir: str, device: str = "cpu"):
    """
    Load a trained Diffusion_BC or Diffusion_QL agent from a results directory.

    Works even if variant.json does NOT contain state_dim/action_dim/max_action
    by inferring them from the env.
    """
    # --- read args / dims ---
    with open(os.path.join(result_dir, "variant.json"), "r") as f:
        cfg = json.load(f)

    algo = cfg["algo"]
    env_name = cfg["env_name"]

    # Try to read dims; if missing, infer from env
    state_dim = cfg.get("state_dim")
    action_dim = cfg.get("action_dim")
    max_action = cfg.get("max_action")

    if (state_dim is None) or (action_dim is None) or (max_action is None):
        # Fallback: create env and infer
        env = gym.make(env_name)
        obs, info = env.reset(seed=cfg.get("seed", 0))
        state_space = env.observation_space
        action_space = env.action_space

        state_dim = state_space.shape[0]
        action_dim = action_space.shape[0]
        max_action = float(action_space.high[0])
        env.close()

    # Hyperparameters (use defaults if missing)
    discount = cfg.get("discount", 0.99)
    tau = cfg.get("tau", 0.005)
    T = cfg.get("T", 5)
    beta_schedule = cfg.get("beta_schedule", "vp")
    lr = cfg.get("lr", 3e-4)
    eta = cfg.get("eta", 1.0)
    max_q_backup = cfg.get("max_q_backup", False)
    lr_decay = cfg.get("lr_decay", False)
    num_epochs = cfg.get("num_epochs", 500)
    gn = cfg.get("gn", 1.0)

    # --- best epoch ---
    with open(os.path.join(result_dir, "best_score.txt"), "r") as f:
        best_info = json.load(f)
    best_epoch = int(best_info["epoch"])

    # --- build agent (same as training) ---
    if algo == "bc":
        agent = Diffusion_BC(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=discount,
            tau=tau,
            beta_schedule=beta_schedule,
            n_timesteps=T,
            lr=lr,
        )
    elif algo == "ql":
        agent = Diffusion_QL(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=discount,
            tau=tau,
            max_q_backup=max_q_backup,
            beta_schedule=beta_schedule,
            n_timesteps=T,
            ema_decay=0.995,
            step_start_ema=1000,
            update_ema_every=5,
            lr=lr,
            lr_decay=lr_decay,
            lr_maxt=num_epochs,
            grad_norm=gn,
        )
    else:
        raise ValueError(f"Unknown algo in variant.json: {algo}")

    # --- load weights for best epoch ---
    agent.load_model(result_dir, best_epoch)
    print(f"Loaded {algo} agent from {result_dir} at epoch {best_epoch}")

    return agent, env_name, algo

def rollout(result_dir: str,
            device: str = "cpu",
            n_episodes: int = 5):
    agent, env_name, algo = load_agent(result_dir, device=device)
    env = gym.make(env_name, render_mode="human")

    returns = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            action = agent.sample_action(np.array(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward

        returns.append(ep_ret)
        print(f"[{algo.upper()}] Episode {ep+1}: return = {ep_ret:.2f}")

    env.close()
    print(f"Average return over {n_episodes} episodes: {np.mean(returns):.2f}")


if __name__ == "__main__":
    # change these paths depending on which policy you want to visualize
    # BC:
    # result_dir = "results/Hopper-v4_exp1_bc"
    # QL:
    # result_dir = "results/Hopper-v4_exp1_bc"
    result_dir = "results/Hopper-v4_exp1_ql"

    device = "cpu"  # or "cuda:0" if you trained & want to run on GPU
    rollout(result_dir, device=device, n_episodes=5)
