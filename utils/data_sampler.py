import torch

class Data_Sampler:
    def __init__(self, data: dict, device: str):
        self.state = torch.from_numpy(data["observations"]).float()
        self.action = torch.from_numpy(data["actions"]).float()
        self.next_state = torch.from_numpy(data["next_observations"]).float()
        reward = torch.from_numpy(data["rewards"]).view(-1, 1).float()

        # support either 'terminals' (D4RL style) or 'dones' (your script)
        if "terminals" in data:
            done_array = data["terminals"]
        elif "dones" in data:
            done_array = data["dones"]
        else:
            raise KeyError("Dataset must contain 'terminals' or 'dones'")

        self.not_done = 1.0 - torch.from_numpy(done_array).view(-1, 1).float()
        self.reward = reward

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]
        self.device = device

    def sample(self, batch_size: int):
        ind = torch.randint(0, self.size, size=(batch_size,))
        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
            self.next_state[ind].to(self.device),
            self.reward[ind].to(self.device),
            self.not_done[ind].to(self.device),
        )
