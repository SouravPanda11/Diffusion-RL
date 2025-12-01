# Copyright 2025 Sourav Panda
# Unified + Simplified main.py for Diffusion Policies
# Only CLI argument needed: --algo {bc, ql}

import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch

from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
from load_and_mix_datasets import load_and_mix_hopper_datasets


# =========================================================================== #
# Training Loop
# =========================================================================== #

def train_agent(state_dim, action_dim, max_action, device, output_dir, args, dataset):

    # Dataset wrapper → PyTorch tensors + sampling
    data_sampler = Data_Sampler(dataset, device)
    utils.print_banner("Loaded offline dataset (mixed Hopper)")

    # ------------------------------------------------------------------ #
    # Instantiate the chosen algorithm
    # ------------------------------------------------------------------ #
    if args.algo == "ql":
        from agents.ql_diffusion import Diffusion_QL as Agent
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=args.discount,
            tau=args.tau,
            max_q_backup=args.max_q_backup,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            eta=args.eta,
            lr=args.lr,
            lr_decay=False,
            lr_maxt=args.num_epochs,
            grad_norm=args.gn,
        )
    elif args.algo == "bc":
        from agents.bc_diffusion import Diffusion_BC as Agent
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=args.discount,
            tau=args.tau,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            lr=args.lr,
        )
    else:
        raise ValueError(f"Unknown algo: {args.algo}")

    # ------------------------------------------------------------------ #
    # Training setup
    # ------------------------------------------------------------------ #
    early_stop = False
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.0)
    writer = None

    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    best_eval_return = -np.inf

    utils.print_banner("Training Start", separator="*", num_star=90)

    # =================================================================== #
    # Main Training Loop
    # =================================================================== #
    while (training_iters < max_timesteps) and (not early_stop):

        iterations = args.eval_freq * args.num_steps_per_epoch

        loss_metric = agent.train(
            data_sampler,
            iterations=iterations,
            batch_size=args.batch_size,
            log_writer=writer,
        )

        training_iters += iterations
        curr_epoch = training_iters // args.num_steps_per_epoch

        # -------------- Logging ----------------
        utils.print_banner(
            f"Train step: {training_iters}", separator="*", num_star=90
        )
        logger.record_tabular("Epochs", curr_epoch)
        logger.record_tabular("BC Loss", np.mean(loss_metric["bc_loss"]))
        logger.record_tabular("QL Loss", np.mean(loss_metric["ql_loss"]))
        logger.record_tabular("Actor Loss", np.mean(loss_metric["actor_loss"]))
        logger.record_tabular("Critic Loss", np.mean(loss_metric["critic_loss"]))
        logger.dump_tabular()

        # -------------- Evaluation ----------------
        avg_ret, std_ret = eval_policy(
            agent,
            env_name=args.env_name,
            seed=args.seed,
            eval_episodes=args.eval_episodes,
        )

        evaluations.append(
            [avg_ret, std_ret,
             np.mean(loss_metric["bc_loss"]),
             np.mean(loss_metric["ql_loss"]),
             np.mean(loss_metric["actor_loss"]),
             np.mean(loss_metric["critic_loss"]),
             curr_epoch]
        )
        np.save(os.path.join(output_dir, "eval.npy"), np.array(evaluations))

        logger.record_tabular("Eval Return", avg_ret)
        logger.dump_tabular()

        # Early stop logic
        bc_loss = np.mean(loss_metric["bc_loss"])
        if args.early_stop:
            early_stop = stop_check(best_eval_return, bc_loss)

        # Checkpoint
        if avg_ret > best_eval_return:
            best_eval_return = avg_ret
            agent.save_model(output_dir, curr_epoch)

    # Save best summary
    scores = np.array(evaluations)
    best_id = np.argmax(scores[:, 0])
    best_res = {
        "epoch": int(scores[best_id, -1]),
        "best raw score avg": float(scores[best_id, 0]),
        "best raw score std": float(scores[best_id, 1]),
    }
    with open(os.path.join(output_dir, "best_score.txt"), "w") as f:
        f.write(json.dumps(best_res))


# =========================================================================== #
# Evaluation Helper (Gymnasium)
# =========================================================================== #

def eval_policy(policy, env_name, seed, eval_episodes=10):
    env = gym.make(env_name)

    scores = []
    for _ in range(eval_episodes):
        obs, _ = env.reset(seed=seed + 100)
        done = False
        total = 0.0

        while not done:
            action = policy.sample_action(np.array(obs))
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            total += r

        scores.append(total)

    env.close()

    avg, std = float(np.mean(scores)), float(np.std(scores))
    utils.print_banner(f"Eval over {eval_episodes} episodes: {avg:.2f} ± {std:.2f}")
    return avg, std


# =========================================================================== #
# Entry Point — ONLY argument needed = algo
# =========================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="ql", type=str, help="bc or ql")
    parser.add_argument("--exp", default="exp1", type=str)
    args = parser.parse_args()

    # Fixed hyperparameters (no need to type in CLI)
    args.env_name = "Hopper-v4"
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.dir = "results"
    args.seed = 0

    # Training settings
    args.batch_size = 256
    args.lr = 3e-4
    args.discount = 0.99
    args.tau = 0.005
    args.T = 5
    args.beta_schedule = "vp"
    args.num_epochs = 500           # you can change default
    args.eval_freq = 20
    args.eval_episodes = 5
    args.num_steps_per_epoch = 1000
    args.eta = 1.0
    args.max_q_backup = False
    args.lr_decay = False
    args.early_stop = False
    args.gn = 1.0
    args.save_best_model = True

    # ---------------------------------------------------------
    # Mixed dataset (random + medium + expert)
    # ---------------------------------------------------------
    dataset = load_and_mix_hopper_datasets()

    # Environment dims
    env = gym.make(args.env_name)
    obs, _ = env.reset(seed=args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    env.close()

    # Logging
    safe_exp_name = f"{args.env_name}_{args.exp}_{args.algo}"
    results_dir = os.path.join(args.dir, safe_exp_name)
    os.makedirs(results_dir, exist_ok=True)

    setup_logger(safe_exp_name, vars(args), log_dir=results_dir)
    utils.print_banner(f"Saving at: {results_dir}")

    # Train
    train_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=args.device,
        output_dir=results_dir,
        args=args,
        dataset=dataset,
    )
