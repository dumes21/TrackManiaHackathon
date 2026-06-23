"""Drive the best trained policy for real — a completion run, not training.

Unlike train_evolution.py, this does NOT mutate or save anything. It loads the
best evolved steering weights and drives with the RULE-BASED throttle (accelerate
on straights, brake into corners), attempting to actually complete the lap.

Setup (see docs/HACKATHON_RUNBOOK.md):
1. Launch TMNF through TMInterface 1.4.3, windowed, behind-car camera.
2. Load the map and put the car on the start line.
3. Run:
       python scripts/drive_trained.py --config config/default.yaml

Each attempt respawns to the start line and drives until finish / stuck / off-track
or the step cap. Press Ctrl+C to stop. Keep any --debug window off the game window.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os

import numpy as np

from Agent import GeneticAgent
from tm_env import TrackmaniaEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--weights", default="best_trackmania_weights.npy")
    parser.add_argument("--attempts", type=int, default=0, help="0 = run until Ctrl+C")
    parser.add_argument("--debug", action="store_true", help="Show the vision window + per-step diagnostics.")
    parser.add_argument(
        "--full-throttle",
        action="store_true",
        help="Pin the gas on (ignore the vision-gated brake). The car keeps moving "
        "even when vision is unsure; it will fly off corners but won't stall.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        raise SystemExit(
            f"[drive] No trained weights at {args.weights}. Train first with "
            f"scripts/train_evolution.py, or pass --weights <file>."
        )

    # --full-throttle pins the gas on; otherwise use the rule throttle
    # (accelerate on straights, brake into corners).
    env = TrackmaniaEnv(
        config_path=args.config,
        debug=args.debug,
        training_throttle=bool(args.full_throttle),
    )
    max_steps = int(env.cfg.get("evolution", {}).get("max_steps", 600))
    print("[drive] Throttle: FULL (pinned)" if args.full_throttle
          else "[drive] Throttle: rule-based (accelerate on straights, brake for corners).")

    agent = GeneticAgent()
    agent.weights = np.load(args.weights)
    print(f"[drive] Loaded trained steering weights from {args.weights}")

    attempt = 0
    try:
        while args.attempts == 0 or attempt < args.attempts:
            attempt += 1
            obs, _ = env.reset()
            done = False
            steps = 0
            print(f"\n[ATTEMPT {attempt}] Driving...")

            while not done and steps < max_steps:
                action = agent.get_action(obs)
                obs, reward, done, _, info = env.step(action)
                steps += 1

            end_reason = info.get("end_reason") or ("max_steps" if not done else "?")
            tag = "🏁 FINISHED" if info.get("finished") else f"ended: {end_reason}"
            print(
                f"-> Attempt {attempt} | {tag} | reached {info.get('race_time_s', 0.0):.2f}s | "
                f"dist {info.get('distance_m', 0.0):.0f}m | CP {info.get('checkpoint_current', 0)}/"
                f"{info.get('checkpoint_target', 0)}"
            )
    except KeyboardInterrupt:
        print("\n[drive] Stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
