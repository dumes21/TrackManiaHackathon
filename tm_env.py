"""TrackMania Gymnasium environment backed by TMInterface."""
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
WALL_HIT_SPEED_KMH = 2.0
WALL_HIT_STEPS = 5
OFFTRACK_CONFIDENCE = 0.002
OFFTRACK_FRONT_CLEAR = 0.08
OFFTRACK_STEPS = 15 # CHANGED: Increased from 8 to 15 to give a 0.75-second grace period for jumps
DOWNHILL_SPEED_FACTOR = 0.40 

class TrackmaniaEnv(gym.Env):
    def __init__(self, config_path="config/default.yaml", connect_timeout_s=10.0, debug=False):
        super().__init__()
        self.cfg = load_config(config_path)
        self.debug = bool(debug)
        self.debug_window = self.cfg.get("runtime", {}).get("debug_window_name", "TMNF Evolution Debug")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.num_rays = NUM_RAYS
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_rays,), dtype=np.float32)

        runtime = self.cfg.get("runtime", {})
        wcfg = self.cfg.get("window", {})
        evo = self.cfg.get("evolution", {})
        ctrl = self.cfg.get("controller", {})

        self.dt_target = 1.0 / max(1.0, float(runtime.get("control_hz", 20)))
        self.w_speed = float(evo.get("speed_weight", 0.05))
        self.w_center = float(evo.get("center_weight", 2.0))
        self.w_forward = float(evo.get("forward_weight", 2.0))
        self.checkpoint_bonus = float(evo.get("checkpoint_bonus", 100.0))
        self.finish_bonus = float(evo.get("finish_bonus", 500.0))
        self.crash_penalty = float(evo.get("crash_penalty", 15.0))
        self.respawn_wait_s = float(evo.get("respawn_wait_s", 4.0))
        self.max_steps = int(evo.get("max_steps", 2000))
        self.min_confidence = float(ctrl.get("min_confidence", 0.01))
        self.max_speed_kmh = float(evo.get("max_speed_kmh", 120))

        self.start_grace_steps = int(1.5 / self.dt_target)
        self._step_count = 0
        self._wall_stopped_steps = 0
        self._slow_on_hill_steps = 0
        self._prev_altitude = None
        self._offtrack_steps = 0
        self._post_recovery_steps = 0
        self._recovery_steps = 0 
        self._start_phase_steps = int(0.6 / self.dt_target)
        self._prev_steer = 0.0

        self.bridge = TMInterfaceBridge(server_name=runtime.get("server_name", "TMInterface0"))
        self.bridge.start()
        if not self.bridge.wait_until_connected(timeout_s=connect_timeout_s):
            raise RuntimeError("Could not connect to TMInterface. Launch TMNF through ModLoader with TMInterface enabled.")

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

        self._last_checkpoint = 0
        self._low_conf_steps = 0
        self._last_vis = None
        self._last_telem = None
        self._distance_m = 0.0
        self._prev_distance = 0.0
        self._prev_pos = None
        self._prev_cp = 0
        self._announced_cp = False

    def _rays_to_obs(self, vis) -> np.ndarray:
        dists = [float(r.distance_frac) for r in vis.rays[:self.num_rays]]
        if len(dists) < self.num_rays:
            dists += [1.0] * (self.num_rays - len(dists))
        return np.clip(np.array(dists, dtype=np.float32), 0.0, 1.0)

    def _render(self, vis) -> None:
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

    def _is_climbing(self, telem) -> bool:
        try:
            current_alt = float(telem.position[1])
            if self._prev_altitude is None:
                self._prev_altitude = current_alt
                return False
            climbing = (current_alt - self._prev_altitude) > 0.02
            self._prev_altitude = current_alt
            return climbing
        except Exception:
            return False

    def _is_downhill(self, telem) -> bool:
        try:
            if self._prev_altitude is None:
                return False
            return (float(telem.position[1]) - self._prev_altitude) < -0.02
        except Exception:
            return False

    def _compute_centre_steer(self, vis, is_narrow=False) -> float:
        error = float(vis.road_center_error)
        
        deadzone = 0.02 if is_narrow else 0.08
        if abs(error) < deadzone:
            return 0.0
            
        multiplier = 0.85 if is_narrow else 0.55
        correction = (error - np.sign(error) * deadzone) * multiplier
        
        return float(np.clip(correction, -0.9, 0.9))

    def _apply_wall_avoidance(self, steer: float, obs: np.ndarray, vis=None) -> float:
        if len(obs) < self.num_rays:
            return steer
            
        left_min = float(np.min(obs[:self.num_rays // 2]))
        right_min = float(np.min(obs[self.num_rays // 2:]))
        
        is_narrow = (left_min < 0.50 and right_min < 0.50)
        center_min = float(np.min(obs[7:13])) 
        
        avoid = 0.0
        avoid_thresh = 0.60 

        if center_min < 0.45 and (left_min > 0.45 or right_min > 0.45):
            left_space = np.mean(obs[:self.num_rays // 2])
            right_space = np.mean(obs[self.num_rays // 2:])
            
            if left_space > right_space:
                avoid += 0.95  
            else:
                avoid -= 0.95  
                
            centre = 0.0 
            blended = steer * 0.10 + avoid * 0.90 
            
        else:
            if left_min < avoid_thresh:
                avoid += ((avoid_thresh - left_min) / avoid_thresh) * 0.9
            if right_min < avoid_thresh:
                avoid -= ((avoid_thresh - right_min) / avoid_thresh) * 0.9
                
            centre = self._compute_centre_steer(vis, is_narrow) if vis is not None else 0.0
            
            if is_narrow:
                blended = centre * 0.80 + avoid * 0.20 
            elif vis is not None and vis.confidence > 0.02:
                blended = steer * 0.20 + avoid * 0.40 + centre * 0.40 
            else:
                blended = steer * 0.35 + avoid * 0.40 + centre * 0.25

        return float(np.clip(blended, -0.85, 0.85))

    def _boost_up_hill(self, steer) -> None:
        for _ in range(int(2.5 / self.dt_target)):
            cmd = ControlCommand(steer=steer, accelerate=True, brake=False)
            self.bridge.set_active(True)
            self.bridge.set_command(cmd)
            time.sleep(self.dt_target)

    def _stop_car(self) -> None:
        for _ in range(int(0.3 / self.dt_target)):
            stop_cmd = ControlCommand(steer=0.0, accelerate=False, brake=True)
            self.bridge.set_active(True)
            self.bridge.set_command(stop_cmd)
            time.sleep(self.dt_target)

    def step(self, action):
        if self._recovery_steps > 0:
            self._recovery_steps -= 1
            
            if self._last_vis is not None:
                current_obs = self._rays_to_obs(self._last_vis)
                left_min = float(np.min(current_obs[:self.num_rays // 2]))
                right_min = float(np.min(current_obs[self.num_rays // 2:]))
                
                steer = 1.0 if left_min < right_min else -1.0
            else:
                steer = 0.0

            self.bridge.set_active(True)
            self.bridge.set_command(ControlCommand(steer=steer, accelerate=False, brake=True))
            time.sleep(self.dt_target)
            
            obs, vis, telem = self._observe()
            
            if self._recovery_steps == 0:
                self._post_recovery_steps = int(0.5 / self.dt_target)
                print("[env] Recovery complete, resuming AI control.")
                
            return obs, -5.0, False, False, self._build_info(telem, "recovering")

        agent_steer = float(np.clip(action[0], -1.0, 1.0))
        agent_gas = float(np.clip(action[1], 0.0, 1.0))
        agent_brake = float(np.clip(action[2], 0.0, 1.0))
        self._step_count += 1

        start_phase = self._step_count <= self._start_phase_steps
        if self._last_vis is not None:
            current_obs = self._rays_to_obs(self._last_vis)
            steer = 0.0 if start_phase else self._apply_wall_avoidance(agent_steer, current_obs, vis=self._last_vis)
        else:
            steer = 0.0

        if self._post_recovery_steps > 0:
            self._post_recovery_steps -= 1
            steer = float(np.clip(steer, -0.20, 0.20))

        steer = 0.75 * self._prev_steer + 0.25 * steer
        self._prev_steer = steer

        fwd_clear = float(self._last_vis.front_clearance) if self._last_vis else 1.0
        climbing = self._is_climbing(self._last_telem) if self._last_telem is not None else False
        going_down = self._is_downhill(self._last_telem) if self._last_telem is not None else False

        if fwd_clear > 0.50:
            gas = 1.0
        elif fwd_clear > 0.30:
            gas = 0.75
        else:
            gas = 0.50

        gas *= max(0.0, min(1.0, agent_gas if agent_gas > 0 else 1.0))

        if self._last_vis is not None:
            centre_err = abs(float(self._last_vis.road_center_error))
            if centre_err > 0.35:
                gas *= 0.55
            elif centre_err > 0.20:
                gas *= 0.75
            if fwd_clear < 0.18 or float(np.min(self._rays_to_obs(self._last_vis))) < 0.12:
                steer = self._compute_centre_steer(self._last_vis)
                gas = min(gas, 0.55)

        if going_down:
            gas *= DOWNHILL_SPEED_FACTOR
            agent_brake = max(agent_brake, 0.5)

        if self._last_telem is not None and self._last_telem.speed_kmh > self.max_speed_kmh:
            gas = 0.0
            agent_brake = 1.0

        should_brake = agent_brake > 0.3
        self.bridge.set_active(True)
        self.bridge.set_command(ControlCommand(steer=steer, accelerate=(gas > 0.1), brake=should_brake))
        time.sleep(self.dt_target)

        obs, vis, telem = self._observe()

        if telem.race_time_ms < 0:
            self._prev_pos = telem.position
            self._prev_altitude = float(telem.position[1])
            return obs, 0.0, False, False, self._build_info(telem, "")

        if not self._announced_cp and telem.checkpoint_target > 0:
            self._announced_cp = True
            print(f"[env] Map has {telem.checkpoint_target} checkpoints")

        pos = telem.position
        if self._prev_pos is not None:
            self._distance_m += float(np.linalg.norm(np.array(pos) - np.array(self._prev_pos)))
        self._prev_pos = pos

        climbing = self._is_climbing(telem)
        if climbing and telem.speed_kmh < 40.0 and self._step_count >= self.start_grace_steps:
            self._slow_on_hill_steps += 1
            if self._slow_on_hill_steps >= 10:
                self._boost_up_hill(steer)
                self._slow_on_hill_steps = 0
        else:
            self._slow_on_hill_steps = 0

        dist_delta = self._distance_m - self._prev_distance
        self._prev_distance = self._distance_m

        wall_dist = float(np.min(obs))
        fwd_clear_r = float(vis.front_clearance)
        speed_reward = telem.speed_kmh * self.w_speed

        if wall_dist < 0.05:
            wall_penalty = -120.0
        elif wall_dist < 0.10:
            wall_penalty = -60.0
        elif wall_dist < 0.20:
            wall_penalty = -20.0
        elif wall_dist < 0.30:
            wall_penalty = -5.0
        else:
            wall_penalty = 0.0

        altitude_reward = 5.0 if climbing else 0.0
        centre_error = abs(float(vis.road_center_error))
        centre_reward = (1.0 - centre_error) * 3.0
        centre_bonus = max(0.0, 1.2 - centre_error * 2.0)

        reward = (
            dist_delta * 3.0
            + speed_reward
            + wall_penalty
            + fwd_clear_r * self.w_forward
            + altitude_reward
            + centre_reward
            + centre_bonus
        )

        done = False
        end_reason = ""

        if telem.checkpoint_current > self._last_checkpoint:
            bonus = self.checkpoint_bonus * (telem.checkpoint_current - self._last_checkpoint)
            reward += bonus
            self._last_checkpoint = telem.checkpoint_current

        if telem.finished:
            reward += self.finish_bonus
            done = True
            end_reason = "finish"

        if not done and self._step_count >= self.start_grace_steps:
            if telem.speed_kmh < WALL_HIT_SPEED_KMH and self._distance_m > 6.0:
                self._wall_stopped_steps += 1
                reward -= 2.0
                
                if self._wall_stopped_steps >= 15: 
                    print("[env] Car stuck! Initiating reverse recovery...")
                    self._recovery_steps = int(1.2 / self.dt_target) 
                    self._wall_stopped_steps = 0
            else:
                self._wall_stopped_steps = 0

        # --- CHANGED: Improved Off-Track Detection ---
        if not done:
            # Removed the front_clearance check. If we see almost no road (confidence < 0.002),
            # we are completely off the track onto grass/scenery.
            offtrack = vis.confidence < OFFTRACK_CONFIDENCE
            
            if offtrack:
                self._low_conf_steps += 1
                reward -= 3.0
                if self._low_conf_steps >= OFFTRACK_STEPS:
                    print("[env] Off-track detected (driving on grass)! Respawning...")
                    self._stop_car()
                    reward -= self.crash_penalty
                    done = True
                    end_reason = "offtrack"
            else:
                self._low_conf_steps = 0

        if not done and self._step_count >= self.max_steps:
            done = True
            end_reason = "max_steps"

        return obs, reward, done, False, self._build_info(telem, end_reason)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bridge.release()
        self.rule.reset()
        self.bridge.respawn()
        self._wait_for_green_light()

        self._last_checkpoint = 0
        self._low_conf_steps = 0
        self._distance_m = 0.0
        self._prev_distance = 0.0
        self._prev_pos = None
        self._prev_altitude = None
        self._prev_cp = 0
        self._step_count = 0
        self._wall_stopped_steps = 0
        self._slow_on_hill_steps = 0
        self._announced_cp = False
        self._offtrack_steps = 0
        self._post_recovery_steps = 0
        self._recovery_steps = 0 
        self._prev_steer = 0.0

        self.bridge.set_active(True)
        obs, _, _ = self._observe()
        return obs, {}

    def _wait_for_green_light(self):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.bridge.get_telemetry().race_time_ms < 0:
                break
            time.sleep(0.05)

        deadline = time.monotonic() + self.respawn_wait_s + 6.0
        while time.monotonic() < deadline:
            if self.bridge.get_telemetry().race_time_ms >= 0:
                break
            time.sleep(0.05)

    def _build_info(self, telem, end_reason):
        return {
            "race_time_s": telem.race_time_ms / 1000.0,
            "distance_m": self._distance_m,
            "checkpoint_current": telem.checkpoint_current,
            "checkpoint_target": telem.checkpoint_target,
            "end_reason": end_reason,
            "finished": telem.finished,
        }

    def close(self):
        try:
            self.bridge.release()
        except Exception:
            pass
        if self.debug:
            cv2.destroyAllWindows()