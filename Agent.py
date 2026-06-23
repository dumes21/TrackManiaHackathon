import numpy as np
import os
import time
from datetime import datetime
from tm_env import TrackmaniaEnv

class GeneticAgent:
    def __init__(self, input_size=20, output_size=1):
        # connects 20 raycasts to 1 output (steering)
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros(output_size)

    def get_action(self, obs):
        output = np.dot(obs, self.weights) + self.bias
        return np.tanh(output) # Caps values smoothly between -1.0 (Left) and 1.0 (Right)

    def mutate(self, mutation_rate=0.1):
        child = GeneticAgent()
        child.weights = self.weights + np.random.randn(*self.weights.shape) * mutation_rate
        # Bias is intentionally left at zero (not mutated): only `weights` is saved,
        # and a constant steering offset would make the car permanently veer.
        child.bias = self.bias.copy()
        return child

def _log_improvement(log_path: str, generation: int, best_reward: float, info: dict) -> None:
    """Append one row recording an improvement, creating a header if new."""
    new_file = not os.path.exists(log_path)
    reached_s = info.get("race_time_s", 0.0)
    distance_m = info.get("distance_m", 0.0)
    checkpoints = f"{info.get('checkpoint_current', 0)}/{info.get('checkpoint_target', 0)}"
    with open(log_path, "a", encoding="utf-8") as f:
        if new_file:
            f.write("timestamp,generation,best_reward,reached_s,distance_m,checkpoints,end_reason\n")
        f.write(
            f"{datetime.now():%Y-%m-%d %H:%M:%S},{generation},{best_reward:.2f},"
            f"{reached_s:.2f},{distance_m:.0f},{checkpoints},{info.get('end_reason', '')}\n"
        )


def train(config_path: str = "config/default.yaml", weights_path: str = "best_trackmania_weights.npy", debug: bool = False):
    """Run the (1+1) evolution-strategy training loop until interrupted.

    The best policy is saved to ``weights_path`` whenever it improves, and
    reloaded from there on startup so training resumes across sessions. Each
    improvement is also appended to a CSV log next to the weights file, so you
    keep a history of when/how-good without piling up weight files.

    If ``debug`` is True, a live OpenCV window shows the vision overlay (road
    mask + rays). Keep that window off the TMNF window or capture breaks.
    """
    print("[START] Spinning up Bytes & Bolts Genetic Steering Engine...")
    env = TrackmaniaEnv(config_path=config_path, debug=debug)

    log_path = os.path.splitext(weights_path)[0] + "_log.csv"

    max_steps = int(env.cfg.get("evolution", {}).get("max_steps", 600))

    best_agent = GeneticAgent()
    # Resume from previously evolved weights if available.
    if os.path.exists(weights_path):
        try:
            best_agent.weights = np.load(weights_path)
            print(f"[RESUME] Loaded saved brain weights from {weights_path}")
        except Exception as exc:
            print(f"[RESUME] Could not load {weights_path}: {exc}")

    best_reward = -float('inf')
    generation = 1

    print("\n=== READY ===")
    print("Keep TrackMania open windowed on your screen. Standby...")
    time.sleep(2)

    try:
        while True:
            current_agent = best_agent.mutate() if generation > 1 else best_agent

            obs, info = env.reset()
            total_reward = 0
            steps = 0
            done = False

            print(f"\n[GEN {generation}] Driving...")

            while not done and steps < max_steps:  # Caps the maximum track run time length
                action = current_agent.get_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

            # Per-generation summary, including progress when the lap did not finish.
            end_reason = info.get("end_reason") or ("max_steps" if not done else "?")
            info["end_reason"] = end_reason  # resolve so the CSV log records max_steps too
            reached_s = info.get("race_time_s", 0.0)
            distance_m = info.get("distance_m", 0.0)
            cp_cur = info.get("checkpoint_current", 0)
            cp_target = info.get("checkpoint_target", 0)
            print(
                f"-> Gen {generation} | reward {total_reward:.1f} | reached {reached_s:.2f}s | "
                f"dist {distance_m:.0f}m | CP {cp_cur}/{cp_target} | ended: {end_reason}"
            )

            # Highscore validation loop
            if total_reward > best_reward and steps > 5:
                best_reward = total_reward
                best_agent = current_agent
                print(f"🌟 NEW LEADERBOARD RECORD! Saving brain weights: {best_reward:.2f}")
                np.save(weights_path, best_agent.weights)
                _log_improvement(log_path, generation, best_reward, info)

            generation += 1

    except KeyboardInterrupt:
        print("\n[STOPPED] Training paused. Save state secured.")
        env.close()


if __name__ == "__main__":
    train()