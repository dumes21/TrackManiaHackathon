import numpy as np
import time
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
        child.bias = self.bias + np.random.randn(*self.bias.shape) * mutation_rate
        return child

print("[START] Spinning up Bytes & Bolts Genetic Steering Engine...")
env = TrackmaniaEnv()

best_agent = GeneticAgent()
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
        
        while not done and steps < 600:  # Caps the maximum track run time length
            action = current_agent.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        print(f"-> Gen {generation} Finished | Survival Score: {total_reward:.2f} frames")

        # Highscore validation loop
        if total_reward > best_reward and steps > 5:
            best_reward = total_reward
            best_agent = current_agent
            print(f"🌟 NEW LEADERBOARD RECORD! Saving brain weights: {best_reward:.2f}")
            np.save("best_trackmania_weights.npy", best_agent.weights)

        generation += 1

except KeyboardInterrupt:
    print("\n[STOPPED] Training paused. Save state secured.")
    env.close()