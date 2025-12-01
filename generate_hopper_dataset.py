import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


def collect_transitions(
    env_id: str,
    model_path: str,
    n_episodes: int,
    max_episode_steps: int | None = None,
    deterministic: bool = False,
):
    """
    Roll out a trained policy and collect an offline dataset.

    Returns:
        dataset: dict with keys:
            'observations', 'actions', 'next_observations',
            'rewards', 'dones'
    """
    env = gym.make(env_id)
    model = PPO.load(model_path)

    observations = []
    actions = []
    next_observations = []
    rewards = []
    dones = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = env.step(action)

            observations.append(obs)
            actions.append(action)
            next_observations.append(next_obs)
            rewards.append(reward)
            # “done” is True if episode really ends (terminated or truncated)
            done_flag = terminated or truncated
            dones.append(done_flag)

            obs = next_obs
            done = done_flag
            steps += 1

            if max_episode_steps is not None and steps >= max_episode_steps:
                break

    env.close()

    dataset = {
        "observations": np.array(observations, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "next_observations": np.array(next_observations, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.bool_),
    }

    return dataset


def save_dataset(dataset: dict, path: str):
    """
    Save dataset as a compressed .npz file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **dataset)
    print(f"Saved dataset to {path}")
    

if __name__ == "__main__":
    ENV_ID = "Hopper-v4"   # change if you’re using a different Hopper version

    # paths to your trained PPO agents
    MODELS = {
        "random": "models/ppo_Hopper-v4_randomish.zip",
        "medium": "models/ppo_Hopper-v4_medium.zip",
        "expert": "models/ppo_Hopper-v4_expert.zip",
    }

    # how many episodes per policy you want
    EPISODES_PER_POLICY = {
        "random": 100,
        "medium": 100,
        "expert": 100,
    }

    OUTPUT_DIR = "offline_datasets"

    for name, model_path in MODELS.items():
        print(f"\nCollecting dataset for policy: {name}")
        dataset = collect_transitions(
            env_id=ENV_ID,
            model_path=model_path,
            n_episodes=EPISODES_PER_POLICY[name],
            max_episode_steps=None,
            deterministic=(name == "expert"),
        )

        out_path = os.path.join(OUTPUT_DIR, f"hopper_{name}.npz")
        save_dataset(dataset, out_path)
