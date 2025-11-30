import os
import argparse
from typing import Dict, Any

import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed


# ────────────────────────── CONFIGS ──────────────────────────

PPO_CONFIGS: Dict[str, Dict[str, Any]] = {
    "randomish": {
        "total_timesteps": int(2e5),
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_steps": 128,
        "batch_size": 64,
        "ent_coef": 0.02,
        "clip_range": 0.2,
    },
    "medium": {
        "total_timesteps": int(5e5),
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_steps": 512,
        "batch_size": 64,
        "ent_coef": 0.005,
        "clip_range": 0.2,
    },
    "expert": {
        "total_timesteps": int(1e6),
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_steps": 2048,
        "batch_size": 64,
        "ent_coef": 0.0,
        "clip_range": 0.2,
    },
}


# ─────────────────────── ENV FACTORY ────────────────────────

def make_env(env_id: str, seed: int = 0):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# ─────────────────────── tqdm CALLBACK ───────────────────────

class TQDMCallback(BaseCallback):
    """
    Displays a tqdm progress bar for PPO training.
    SB3 calls _on_step() after every rollout step or mini-batch update.
    """

    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", smoothing=0.1)

    def _on_step(self) -> bool:
        # Increment based on self.model.num_timesteps (SB3 internal tracking)
        if self.pbar is not None:
            self.pbar.n = self.model.num_timesteps
            self.pbar.refresh()
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.n = self.total_timesteps
            self.pbar.refresh()
            self.pbar.close()


# ───────────────────────── TRAINING ─────────────────────────

def train_ppo(env_id: str, config_name: str, seed: int = 0, save_dir: str = "models"):
    assert config_name in PPO_CONFIGS, f"Unknown config '{config_name}'"
    cfg = PPO_CONFIGS[config_name]

    os.makedirs(save_dir, exist_ok=True)
    set_random_seed(seed)

    env = DummyVecEnv([make_env(env_id, seed)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        ent_coef=cfg["ent_coef"],
        clip_range=cfg["clip_range"],
        verbose=0,  # tqdm replaces SB3 logging
        tensorboard_log=os.path.join(save_dir, "tb_logs", config_name),
        seed=seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=int(1e5),
        save_path=os.path.join(save_dir, f"checkpoints_{config_name}"),
        name_prefix="ppo_hopper",
    )

    tqdm_callback = TQDMCallback(total_timesteps=cfg["total_timesteps"])

    print(f"\n=== Training PPO ({config_name}) on {env_id} ===\n")

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=[checkpoint_callback, tqdm_callback],
        progress_bar=False,   # we use our own tqdm
    )

    model_path = os.path.join(save_dir, f"ppo_{env_id}_{config_name}.zip")
    model.save(model_path)
    print(f"\nSaved model to: {model_path}\n")

    env.close()


# ────────────────────────── MAIN ───────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="Hopper-v4")
    parser.add_argument("--config", type=str, default="medium",
                        help=f"One of {list(PPO_CONFIGS.keys())}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ppo(
        env_id=args.env_id,
        config_name=args.config,
        seed=args.seed,
        save_dir=args.save_dir,
    )
