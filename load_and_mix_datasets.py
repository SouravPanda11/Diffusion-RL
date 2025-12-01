import numpy as np

def load_and_mix_hopper_datasets() -> dict:
    paths = [
        "offline_datasets/hopper_random.npz",
        "offline_datasets/hopper_medium.npz",
        "offline_datasets/hopper_expert.npz",
    ]
    arrays = [np.load(p) for p in paths]

    dataset = {}
    for key in ["observations", "actions", "next_observations", "rewards", "dones"]:
        dataset[key] = np.concatenate([a[key] for a in arrays], axis=0)

    return dataset
