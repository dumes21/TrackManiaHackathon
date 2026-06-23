"""Create a dummy `weights.pt` matching `model.RaycastPolicy` shapes.

Run this to produce `weights.pt` so `debbiesagent` can load it for smoke tests.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import RaycastPolicy
import torch


def main():
    model = RaycastPolicy(obs_dim=83, hidden=256)
    sd = model.state_dict()
    # reinitialize with random weights (same shapes)
    new = {}
    for k, v in sd.items():
        new[k] = torch.randn_like(v)
    torch.save(new, "weights.pt")
    print("weights.pt written (dummy random weights).")


if __name__ == "__main__":
    main()
