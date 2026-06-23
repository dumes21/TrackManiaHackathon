import torch
import torch.nn as nn

class RaycastPolicy(nn.Module):
    def __init__(self, obs_dim=83, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)
