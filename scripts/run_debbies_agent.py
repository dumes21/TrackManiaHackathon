"""Run the debbies agent live through TMInterface.

This script connects to TMInterface, captures the game view, computes the vision
features and telemetry, builds a fixed-length observation vector (padding with
zeros if necessary), queries `debbiesagent.Agent.act()` and forwards the result
to the game via TMInterface.

Usage:
  python scripts/run_debbies_agent.py --config config/default.yaml

Notes:
- The agent expects an input vector of length `OBS_DIM` (83 by default). If the
  live features assembled here are shorter they will be zero-padded; if longer
  they'll be truncated.
"""

from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
import numpy as np
import cv2

from tm_live_agent.config import load_config
from tm_live_agent.tminterface_bridge import TMInterfaceBridge, ControlCommand
from tm_live_agent.window_capture import WindowCapture
from tm_live_agent.vision import RoadVision
from tm_live_agent.hotkeys import HotkeyToggle

from debbiesagent import Agent as DebbiesAgent

OBS_DIM = 83
WEIGHT_FILE = "weights.pt"


def build_observation(vis, telem, obs_dim=OBS_DIM):
    # Basic features: ray distances
    ray_dists = np.array([float(r.distance_frac) for r in vis.rays], dtype=np.float32)

    # A small set of extra features from vision + telemetry
    extras = np.array([
        vis.road_center_error,
        vis.best_ray_error,
        vis.front_clearance,
        vis.confidence,
        float(telem.speed_kmh),
        float(telem.checkpoint_current),
        float(telem.checkpoint_target),
    ], dtype=np.float32)

    # Telemetry position/velocity/ypr if available
    try:
        pos = np.array(telem.position, dtype=np.float32)
        vel = np.array(telem.velocity, dtype=np.float32)
        ypr = np.array(telem.yaw_pitch_roll, dtype=np.float32)
    except Exception:
        pos = np.zeros(3, dtype=np.float32)
        vel = np.zeros(3, dtype=np.float32)
        ypr = np.zeros(3, dtype=np.float32)

    parts = [ray_dists, extras, pos, vel, ypr]
    flat = np.concatenate([p.flatten() for p in parts], axis=0).astype(np.float32)

    if len(flat) < obs_dim:
        pad = np.zeros(obs_dim - len(flat), dtype=np.float32)
        flat = np.concatenate([flat, pad], axis=0)
    elif len(flat) > obs_dim:
        flat = flat[:obs_dim]

    # Agent expects a list of arrays which it concatenates internally; give one array.
    return [flat]


def map_output_to_command(out_arr: np.ndarray) -> ControlCommand:
    steer = float(np.clip(out_arr[0], -1.0, 1.0))
    gas = float(np.clip(out_arr[1], 0.0, 1.0))
    brake = float(np.clip(out_arr[2], 0.0, 1.0))

    # TMInterface client uses booleans for accelerate/brake. Use thresholds.
    accel_bool = gas > 0.45
    brake_bool = brake > 0.45

    return ControlCommand(steer=steer, accelerate=accel_bool, brake=brake_bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runtime = cfg.get("runtime", {})
    wcfg = cfg.get("window", {})

    print("[debbies] Starting TMInterface bridge...")
    bridge = TMInterfaceBridge(server_name=runtime.get("server_name", "TMInterface0"))
    bridge.start()
    if not bridge.wait_until_connected(timeout_s=10):
        raise RuntimeError("Could not connect to TMInterface. Launch TMNF through TMInterface 1.4.3 first.")

    print("[debbies] Locating TMNF window...")
    capture = WindowCapture(
        title_substring=wcfg.get("title_substring", "TmForever"),
        crop=(
            float(wcfg.get("crop_left", 0.0)),
            float(wcfg.get("crop_top", 0.0)),
            float(wcfg.get("crop_right", 1.0)),
            float(wcfg.get("crop_bottom", 1.0)),
        ),
    )

    vision = RoadVision(cfg)
    hotkeys = HotkeyToggle(start_key=runtime.get("start_key", "F8"), stop_key=runtime.get("stop_key", "ESC"))

    agent = DebbiesAgent()

    control_hz = float(runtime.get("control_hz", 20))
    dt_target = 1.0 / max(1.0, control_hz)

    active = False
    bridge.set_active(False)
    bridge.set_command(ControlCommand())

    print("[debbies] Ready. Press F8 to start/pause. Press ESC for emergency stop.")

    try:
        while True:
            loop_start = time.time()

            if hotkeys.stop_pressed():
                print("[debbies] ESC pressed: emergency stop.")
                active = False
                bridge.release()
                break

            if hotkeys.start_pressed():
                active = not active
                bridge.set_active(active)
                print(f"[debbies] {'ACTIVE' if active else 'PAUSED'}")

            frame = capture.grab()
            vis = vision.process(frame)
            telem = bridge.get_telemetry()

            obs = build_observation(vis, telem, obs_dim=OBS_DIM)

            if active:
                out = agent.act(obs)
                cmd = map_output_to_command(out)
                bridge.set_active(True)
                bridge.set_command(cmd)
            else:
                bridge.set_command(ControlCommand())

            if not args.no_debug:
                view = vis.debug_bgr
                cv2.imshow("debbies-debug", view)
                cv2.waitKey(1)

            elapsed = time.time() - loop_start
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)
    finally:
        bridge.release()
        cv2.destroyAllWindows()
        print("[debbies] Inputs released. Done.")


if __name__ == "__main__":
    main()
