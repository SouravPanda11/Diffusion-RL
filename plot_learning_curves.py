# plot_learning_curves.py
import os
import numpy as np
import matplotlib.pyplot as plt


def load_eval(result_dir: str):
    path = os.path.join(result_dir, "eval.npy")
    arr = np.load(path)
    # columns: [avg_ret, std_ret, bc_loss, ql_loss,
    #           actor_loss, critic_loss, epoch]
    return arr


def plot_single_run(result_dir: str, label: str):
    evals = load_eval(result_dir)
    epochs = evals[:, 6]
    avg_ret = evals[:, 0]
    bc_loss = evals[:, 2]
    ql_loss = evals[:, 3]

    # --- figure 1: return ---
    plt.figure()
    plt.plot(epochs, avg_ret)
    plt.xlabel("Epoch")
    plt.ylabel("Average episodic return")
    plt.title(f"Return vs Epoch ({label})")
    plt.grid(True)
    plt.tight_layout()

    # --- figure 2: losses ---
    plt.figure()
    plt.plot(epochs, bc_loss, label="BC loss")
    plt.plot(epochs, ql_loss, label="QL loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Losses vs Epoch ({label})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


if __name__ == "__main__":
    # adjust these to your folders
    bc_dir = "results/Hopper-v4_exp1_bc"
    ql_dir = "results/Hopper-v4_exp1_ql"

    # BC curves
    plot_single_run(bc_dir, label="BC")

    # QL curves
    plot_single_run(ql_dir, label="QL")

    plt.show()
