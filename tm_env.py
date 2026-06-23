"""TrackMania Gymnasium environment backed by TMInterface.

This is the merged version of the original screen-capture/keyboard env. Instead of
``pydirectinput`` and guessing speed from screen motion, it uses:

- the TMInterface bridge for clean input injection and real telemetry
  (speed, checkpoints, finish),
- RoadVision pseudo-LIDAR for the 20-ray observation,
- the RuleController for HYBRID speed control: the agent (genetic policy)
  supplies steering, while throttle/brake come from the rule controller's
  real-telemetry, corner-aware logic.

The observation and action spaces match the original env so the genetic
``GeneticAgent`` (20 inputs -> 1 steering output) and its saved weights still work.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import time

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tm_live_agent.config import load_config
from tm_live_agent.controller import RuleController
from tm_live_agent.tminterface_bridge import ControlCommand, TMInterfaceBridge
from tm_live_agent.vision import RoadVision
from tm_live_agent.window_capture import WindowCapture

NUM_RAYS = 20


class TrackmaniaEnv(gym.Env):
    def __init__(self, config_path: str = "config/default.yaml", connect_timeout_s: float = 10.0, debug: bool = False):
        super().__init__()

        self.cfg = load_config(config_path)
        self.debug = bool(debug)
        self.debug_window = self.cfg.get("runtime", {}).get("debug_window_name", "TMNF Evolution Debug")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.num_rays = NUM_RAYS
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_rays,), dtype=np.float32)

        runtime = self.cfg.get("runtime", {})
        wcfg = self.cfg.get("window", {})
        evo = self.cfg.get("evolution", {})

        # Real-time control step, derived from control_hz (matches run_live_agent).
        self.dt_target = 1.0 / max(1.0, float(runtime.get("control_hz", 20)))

        # Reward / termination shaping.
        self.w_speed = float(evo.get("speed_weight", 0.02))
        self.w_center = float(evo.get("center_weight", 2.0))
        self.w_forward = float(evo.get("forward_weight", 1.0))
        self.checkpoint_bonus = float(evo.get("checkpoint_bonus", 50.0))
        self.finish_bonus = float(evo.get("finish_bonus", 500.0))
        self.crash_penalty = float(evo.get("crash_penalty", 25.0))
        self.offtrack_patience = int(evo.get("offtrack_confidence_patience", 20))
        self.respawn_wait_s = float(evo.get("respawn_wait_s", 4.0))
        self.min_confidence = float(self.cfg.get("controller", {}).get("min_confidence", 0.035))

        # Connect the bridge (input + telemetry).
        self.bridge = TMInterfaceBridge(server_name=runtime.get("server_name", "TMInterface0"))
        self.bridge.start()
        if not self.bridge.wait_until_connected(timeout_s=connect_timeout_s):
            raise RuntimeError(
                "Could not connect to TMInterface. Launch TMNF through TMInterface 1.4.3 first."
            )

        # Perception + hybrid speed controller (reused from the live agent).
        self.capture = WindowCapture(
            title_substring=wcfg.get("title_substring", "TmForever"),
            crop=(
                float(wcfg.get("crop_left", 0.0)),
                float(wcfg.get("crop_top", 0.0)),
                float(wcfg.get("crop_right", 1.0)),
                float(wcfg.get("crop_bottom", 1.0)),
            ),
        )
        self.vision = RoadVision(self.cfg)
        self.rule = RuleController(self.cfg)

        # Per-episode tracking.
        self._last_checkpoint = 0
        self._low_conf_steps = 0
        self._last_vis = None
        self._last_telem = None
        self._distance_m = 0.0
        self._prev_pos = None
        self._announced_cp = False  # one-time diagnostic when checkpoints first register

    def _rays_to_obs(self, vis) -> np.ndarray:
        """Take ray distances as a fixed-length [0,1] observation (clamp/pad to num_rays)."""
        dists = [float(r.distance_frac) for r in vis.rays[: self.num_rays]]
        if len(dists) < self.num_rays:
            # Pad with 1.0 (treat missing rays as "open") so the policy size is stable.
            dists += [1.0] * (self.num_rays - len(dists))
        return np.clip(np.array(dists, dtype=np.float32), 0.0, 1.0)

    def _render(self, vis) -> None:
        """Show the OpenCV vision overlay (road mask + rays). Off unless --debug.

        Keep this window OFF the TMNF window, or the capture will photograph it.
        """
        if not self.debug:
            return
        view = vis.debug_bgr
        if view.shape[1] > 900:
            scale = 900 / view.shape[1]
            view = cv2.resize(view, (900, int(view.shape[0] * scale)))
        cv2.imshow(self.debug_window, view)
        cv2.waitKey(1)

    def _observe(self):
        frame = self.capture.grab()
        vis = self.vision.process(frame)
        telem = self.bridge.get_telemetry()
        self._last_vis = vis
        self._last_telem = telem
        self._render(vis)
        return self._rays_to_obs(vis), vis, telem

    def step(self, action):
        steer = float(np.clip(action[0], -1.0, 1.0))

        # Don't let the rule controller's stuck timer count the start countdown:
        # only treat the car as "active" once the race clock is non-negative.
        pre_telem = self._last_telem
        racing = pre_telem is not None and pre_telem.race_time_ms >= 0

        # Hybrid: rule controller decides throttle/brake from real telemetry; the
        # genetic policy overrides only the steering.
        cmd, ctrl_dbg = self.rule.compute(self._last_vis, pre_telem, active=racing)
        cmd = ControlCommand(steer=steer, accelerate=cmd.accelerate, brake=cmd.brake)
        self.bridge.set_active(True)
        self.bridge.set_command(cmd)

        time.sleep(self.dt_target)

        obs, vis, telem = self._observe()

        # During the start countdown (negative race clock) the car is frozen on the
        # line — hold position and neither score nor terminate the episode.
        if telem.race_time_ms < 0:
            self._prev_pos = telem.position
            return obs, 0.0, False, False, self._build_info(telem, end_reason="")

        # One-time diagnostic: confirm whether this map exposes checkpoints/finish.
        if not self._announced_cp and telem.checkpoint_target > 0:
            self._announced_cp = True
            print(f"[env] Checkpoints registered for this map: target={telem.checkpoint_target}")

        # Accumulate distance travelled (metres) from telemetry position.
        pos = telem.position
        if self._prev_pos is not None:
            self._distance_m += float(np.linalg.norm(np.array(pos) - np.array(self._prev_pos)))
        self._prev_pos = pos

        wall_distance = float(np.min(obs))
        forward_vision = float(vis.front_clearance)

        reward = (
            telem.speed_kmh * self.w_speed
            + wall_distance * self.w_center
            + forward_vision * self.w_forward
        )

        done = False
        end_reason = ""

        # Checkpoint progress bonus.
        if telem.checkpoint_current > self._last_checkpoint:
            reward += self.checkpoint_bonus * (telem.checkpoint_current - self._last_checkpoint)
            self._last_checkpoint = telem.checkpoint_current

        # Finish -> big reward, end episode.
        if telem.finished:
            reward += self.finish_bonus
            done = True
            end_reason = "finish"
            print(f"🏁 [FINISH] race_time={telem.race_time_ms / 1000:.2f}s")

        # Stuck (real telemetry, via the rule controller's stuck detector).
        if not done and ctrl_dbg.stuck_seconds >= float(self.rule.cfg.get("stuck_after_seconds", 1.6)):
            reward -= self.crash_penalty
            done = True
            end_reason = "stuck"
            print("💥 [STUCK] low speed sustained — resetting.")

        # Off-track: vision confidence collapsed for a sustained window.
        if not done:
            if vis.confidence < self.min_confidence:
                self._low_conf_steps += 1
            else:
                self._low_conf_steps = 0
            if self._low_conf_steps >= self.offtrack_patience:
                reward -= self.crash_penalty
                done = True
                end_reason = "offtrack"
                print("🌫️ [OFF-TRACK] lost the road — resetting.")

        return obs, reward, done, False, self._build_info(telem, end_reason)

    def _build_info(self, telem, end_reason: str) -> dict:
        return {
            "race_time_s": telem.race_time_ms / 1000.0,
            "distance_m": self._distance_m,
            "checkpoint_current": telem.checkpoint_current,
            "checkpoint_target": telem.checkpoint_target,
            "end_reason": end_reason,
            "finished": telem.finished,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.bridge.release()
        self.rule.reset()
        self.bridge.respawn()
        self._wait_for_green_light()

        self._last_checkpoint = 0
        self._low_conf_steps = 0
        self._distance_m = 0.0
        self._prev_pos = None

        self.bridge.set_active(True)
        obs, _, _ = self._observe()
        return obs, {}

    def _wait_for_green_light(self) -> None:
        """Block until the start countdown ends (race clock reaches 0).

        First wait for the respawn to take effect — the race clock goes negative
        during the countdown — then wait for the green light. This stops the
        agent being evaluated while the car is frozen on the start line.
        """
        # Phase 1: respawn takes effect (clock goes negative = countdown running).
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.bridge.get_telemetry().race_time_ms < 0:
                break
            time.sleep(0.05)
        # Phase 2: wait for the green light (clock reaches 0+).
        deadline = time.monotonic() + self.respawn_wait_s + 6.0
        while time.monotonic() < deadline:
            if self.bridge.get_telemetry().race_time_ms >= 0:
                break
            time.sleep(0.05)

    def close(self):
        try:
            self.bridge.release()
        except Exception:
            pass
        if self.debug:
            cv2.destroyAllWindows()
