# visualize_ppo.py
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


ENV_ID = "Hopper-v4"
MODEL_PATH = "models/ppo_Hopper-v4_randomish.zip"   # change to random/medium as needed
N_EPISODES = 5
DETERMINISTIC = True


def main():
    env = gym.make(ENV_ID, render_mode="human")
    model = PPO.load(MODEL_PATH)

    for ep in range(N_EPISODES):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward

        print(f"[PPO] Episode {ep+1}: return = {ep_ret:.2f}")

    env.close()


if __name__ == "__main__":
    main()
