from __future__ import annotations
import numpy as np
import os
from datetime import datetime

WEIGHTS_PATH = "best_trackmania_weights.npy"
INPUT_SIZE = 20

class Agent:
    def __init__(self):
        self.weights = np.random.randn(INPUT_SIZE, 3) * 0.05
        self.bias = np.zeros(3)
        if os.path.exists(WEIGHTS_PATH):
            try:
                data = np.load(WEIGHTS_PATH, allow_pickle=True).item()
                self.weights = data["weights"]
                self.bias = data["bias"]
                print(f"[Agent] Loaded weights from {WEIGHTS_PATH}")
            except Exception as e:
                print(f"[Agent] Could not load weights: {e}")
        else:
            print("[Agent] No weights found — run train_evolution.py first.")

    def act(self, observation) -> np.ndarray:
        obs_in = np.array(observation).flatten()[:INPUT_SIZE].astype(np.float32)
        if len(obs_in) < INPUT_SIZE:
            obs_in = np.pad(obs_in, (0, INPUT_SIZE - len(obs_in)))
        out = np.tanh(obs_in @ self.weights + self.bias)
        steer = float(np.clip(out[0], -0.55, 0.55))
        gas = float(np.clip(out[1] * 0.25 + 0.75, 0.45, 0.90))
        brake = float(np.clip(out[2] * 0.20, 0.0, 0.40))
        return np.array([steer, gas, brake], dtype=np.float32)

class GeneticAgent:
    def __init__(self):
        self.weights = np.random.randn(INPUT_SIZE, 3) * 0.05
        self.bias = np.zeros(3)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_in = obs.flatten()[:INPUT_SIZE].astype(np.float32)
        if len(obs_in) < INPUT_SIZE:
            obs_in = np.pad(obs_in, (0, INPUT_SIZE - len(obs_in)))
        out = np.tanh(obs_in @ self.weights + self.bias)
        steer = float(np.clip(out[0], -0.55, 0.55))
        gas = float(np.clip(out[1] * 0.25 + 0.75, 0.45, 0.90))
        brake = float(np.clip(out[2] * 0.20, 0.0, 0.40))
        return np.array([steer, gas, brake], dtype=np.float32)

    def mutate(self, mutation_rate=0.15) -> "GeneticAgent":
        child = GeneticAgent()
        child.weights = self.weights + np.random.randn(*self.weights.shape) * mutation_rate
        child.bias = self.bias + np.random.randn(*self.bias.shape) * mutation_rate
        return child

def _log(log_path, generation, best_reward, info):
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if new_file:
            f.write("timestamp,generation,best_reward,race_time_s,distance_m,checkpoints,end_reason\n")
        f.write(
            f"{datetime.now():%Y-%m-%d %H:%M:%S},{generation},{best_reward:.2f},"
            f"{info.get('race_time_s',0):.2f},{info.get('distance_m',0):.0f},"
            f"{info.get('checkpoint_current',0)}/{info.get('checkpoint_target',0)},"
            f"{info.get('end_reason','')}\n"
        )

def train(config_path="config/default.yaml", weights_path=WEIGHTS_PATH, debug=False):
    from tm_env import TrackmaniaEnv

    print("[START] Genetic driver starting...")
    env = TrackmaniaEnv(config_path=config_path, debug=debug)
    log_path = os.path.splitext(weights_path)[0] + "_log.csv"
    max_steps = int(env.cfg.get("evolution", {}).get("max_steps", 3000))
    mut_rate = float(env.cfg.get("evolution", {}).get("mutation_rate", 0.15))

    best_agent = GeneticAgent()
    if os.path.exists(weights_path):
        try:
            data = np.load(weights_path, allow_pickle=True).item()
            best_agent.weights = data["weights"]
            best_agent.bias = data["bias"]
            print(f"[RESUME] Loaded weights from {weights_path}")
        except Exception as e:
            print(f"[WARN] Could not load: {e}")

    best_reward = -float("inf")
    generation = 1
    no_improve_count = 0

    print("\n=== READY — Keep TrackMania open on the start line ===")

    try:
        while True:
            rate = mut_rate * (2.0 if no_improve_count > 8 else 1.0)
            current = best_agent.mutate(mutation_rate=rate) if generation > 1 else best_agent

            print(f"\n[GEN {generation}] driving... (mutation={rate:.3f})")
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0
            done = False

            while not done and steps < max_steps:
                action = current.get_action(obs)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                steps += 1

            end = info.get('end_reason', '?')
            print(
                f"→ Gen {generation} | reward {total_reward:.1f} | "
                f"dist {info.get('distance_m',0):.0f}m | "
                f"CP {info.get('checkpoint_current',0)}/{info.get('checkpoint_target',0)} | "
                f"{end}"
            )

            if total_reward > best_reward and steps > 30 and end != "offtrack":
                best_reward = total_reward
                best_agent = current
                no_improve_count = 0
                print(f"⭐ NEW BEST {best_reward:.1f} — saving")
                np.save(weights_path, {'weights': best_agent.weights, 'bias': best_agent.bias})
                _log(log_path, generation, best_reward, info)
            else:
                no_improve_count += 1

            generation += 1

    except KeyboardInterrupt:
        print("\n[STOPPED]")
        env.close()