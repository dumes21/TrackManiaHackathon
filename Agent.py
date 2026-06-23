from __future__ import annotations
import numpy as np
import os
import time
from datetime import datetime

WEIGHTS_PATH = "best_trackmania_weights.npy"
INPUT_SIZE   = 20


class Agent:
    """Used by run_agent.py for live inference."""
    def __init__(self):
        self.weights = np.random.randn(INPUT_SIZE, 1) * 0.1
        self.bias    = np.zeros(1)
        if os.path.exists(WEIGHTS_PATH):
            try:
                self.weights = np.load(WEIGHTS_PATH)
                print(f"[Agent] Loaded weights from {WEIGHTS_PATH}")
            except Exception as e:
                print(f"[Agent] Could not load weights: {e}. Using random init.")
        else:
            print(f"[Agent] WARNING: {WEIGHTS_PATH} not found — run train_evolution.py first.")

    def act(self, observation) -> np.ndarray:
        flat = np.concatenate(
            [np.array(o).flatten() for o in observation]
        ).astype(np.float32)

        obs_in = flat[:INPUT_SIZE]
        if len(obs_in) < INPUT_SIZE:
            obs_in = np.pad(obs_in, (0, INPUT_SIZE - len(obs_in)))

        steer = float(np.tanh(np.dot(obs_in, self.weights) + self.bias)[0])

        front = float(obs_in[0])
        if front > 0.3:
            gas, brake = 1.0, 0.0    # full throttle unless wall is very close
        elif front > 0.15:
            gas, brake = 0.5, 0.0
        else:
            gas, brake = 0.0, 1.0

        return np.array([steer, gas, brake], dtype=np.float32)


class GeneticAgent:
    def __init__(self):
        self.weights = np.random.randn(INPUT_SIZE, 1) * 0.1
        self.bias    = np.zeros(1)

    def get_action(self, obs: np.ndarray) -> float:
        obs_in = obs.flatten()[:INPUT_SIZE]
        if len(obs_in) < INPUT_SIZE:
            obs_in = np.pad(obs_in, (0, INPUT_SIZE - len(obs_in)))
        return float(np.tanh(np.dot(obs_in, self.weights) + self.bias)[0])

    def mutate(self, mutation_rate=0.3) -> "GeneticAgent":  # increased from 0.1
        child = GeneticAgent()
        child.weights = self.weights + np.random.randn(*self.weights.shape) * mutation_rate
        child.bias    = self.bias.copy()
        return child


def _log_improvement(log_path, generation, best_reward, info):
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if new_file:
            f.write("timestamp,generation,best_reward,race_time_s,distance_m,checkpoints,end_reason\n")
        f.write(
            f"{datetime.now():%Y-%m-%d %H:%M:%S},{generation},{best_reward:.2f},"
            f"{info.get('race_time_s', 0):.2f},{info.get('distance_m', 0):.0f},"
            f"{info.get('checkpoint_current', 0)}/{info.get('checkpoint_target', 0)},"
            f"{info.get('end_reason', '')}\n"
        )


def train(config_path="config/default.yaml",
          weights_path=WEIGHTS_PATH,
          debug=False):
    from tm_env import TrackmaniaEnv
    print("[START] Genetic Steering Engine starting...")
    env = TrackmaniaEnv(config_path=config_path, debug=debug)
    log_path = os.path.splitext(weights_path)[0] + "_log.csv"
    max_steps = int(env.cfg.get("evolution", {}).get("max_steps", 600))

    best_agent = GeneticAgent()
    if os.path.exists(weights_path):
        try:
            best_agent.weights = np.load(weights_path)
            print(f"[RESUME] Loaded weights from {weights_path}")
        except Exception as e:
            print(f"[WARN] Could not load weights: {e}")

    best_reward = -float("inf")
    generation  = 1
    print("\n=== READY — Keep TrackMania open windowed ===")
    time.sleep(2)

    try:
        while True:
            current_agent = best_agent.mutate() if generation > 1 else best_agent
            obs, info = env.reset()
            total_reward = 0.0
            steps        = 0
            done         = False
            print(f"\n[GEN {generation}] Driving...")

            while not done and steps < max_steps:
                steer = current_agent.get_action(obs)
                front = float(obs.flatten()[0]) if len(obs.flatten()) > 0 else 1.0

                # Force full throttle for first 60 steps (~3 sec) to get moving
                if steps < 60:
                    gas, brake = 1.0, 0.0
                elif front > 0.3:
                    gas, brake = 1.0, 0.0
                elif front > 0.15:
                    gas, brake = 0.5, 0.0
                else:
                    gas, brake = 0.0, 1.0

                action = np.array([steer, gas, brake], dtype=np.float32)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

            end_reason = info.get("end_reason") or ("max_steps" if not done else "?")
            info["end_reason"] = end_reason
            dist = info.get('distance_m', 0)
            print(
                f"-> Gen {generation} | reward {total_reward:.1f} | "
                f"time {info.get('race_time_s', 0):.2f}s | "
                f"dist {dist:.0f}m | "
                f"CP {info.get('checkpoint_current', 0)}/{info.get('checkpoint_target', 0)} | "
                f"ended: {end_reason}"
            )

            if total_reward > best_reward and steps > 5:
                best_reward = total_reward
                best_agent  = current_agent
                print(f"NEW BEST: {best_reward:.2f} — saving weights")
                np.save(weights_path, best_agent.weights)
                _log_improvement(log_path, generation, best_reward, info)

            generation += 1

    except KeyboardInterrupt:
        print("\n[STOPPED] Best weights already saved.")
        env.close()
        