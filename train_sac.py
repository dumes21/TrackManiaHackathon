import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tmrl
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
from model import RaycastPolicy

OBS_DIM   = 83
SAVE_DIR  = "./checkpoints"
FINAL_OUT = "./weights_sac"

class TmnfFlatWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0, 1.0], dtype=np.float32),
        )

    def _flatten_obs(self, obs):
        return np.concatenate(
            [np.array(o).flatten() for o in obs], axis=0
        ).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info

def make_env():
    raw_env = gym.make("real-time-gym-v1")
    return TmnfFlatWrapper(raw_env)

if __name__ == "__main__":
    env = make_env()
    check_env(env, warn=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=5_000,
        save_path=SAVE_DIR,
        name_prefix="sac_tmnf",
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    model.learn(
        total_timesteps=200_000,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save(FINAL_OUT)

    actor_state_dict = model.policy.actor.state_dict()
    torch.save(actor_state_dict, "weights.pt")
    print("Done — weights.pt saved")
    env.close()
