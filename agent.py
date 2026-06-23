import torch
import numpy as np
from model import RaycastPolicy

OBS_DIM = 83
WEIGHTS = "weights.pt"

class Agent:
    def __init__(self):
        self.policy = RaycastPolicy(obs_dim=OBS_DIM, hidden=256)
        self.policy.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        self.policy.eval()

    def act(self, observation) -> np.ndarray:
        flat = np.concatenate(
            [np.array(o).flatten() for o in observation], axis=0
        ).astype(np.float32)
        x = torch.FloatTensor(flat).unsqueeze(0)
        with torch.no_grad():
            raw = self.policy(x).squeeze(0).numpy()
        steer = float(np.clip(raw[0], -1.0,  1.0))
        gas   = float(np.clip(raw[1],  0.0,  1.0))
        brake = float(np.clip(raw[2],  0.0,  1.0))
        return np.array([steer, gas, brake], dtype=np.float32)
