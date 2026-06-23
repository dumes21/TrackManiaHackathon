"""Train the genetic-algorithm driver (neuroevolution) on the live game.

Wraps Agent.py with a --config flag, consistent with the other scripts.

Setup (see docs/HACKATHON_RUNBOOK.md):
1. Launch TMNF through TMInterface 1.4.3, windowed, behind-car camera.
2. Load a map and put the car on the start line.
3. Run:
       python scripts/train_evolution.py --config config/default.yaml

The agent evolves steering; the rule controller handles throttle/brake.
Best weights are saved to --weights and reloaded on the next run (resumes).
Press Ctrl+C to stop; the best weights are already saved.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from Agent import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--weights", default="best_trackmania_weights.npy")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show a live OpenCV window of what the agent sees (keep it off the TMNF window).",
    )
    args = parser.parse_args()

    train(config_path=args.config, weights_path=args.weights, debug=args.debug)


if __name__ == "__main__":
    main()
