"""Run the live unseen-track agent.

Expected setup:
1. TMNF launched through TMInterface 1.4.3.
2. Game windowed at 1280x960, taskbar auto-hidden.
3. Map loaded, car on start line, behind-car camera.
4. Run this script.
5. Press F8 to start/pause autonomous control. Press ESC for emergency stop.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

import cv2

from tm_live_agent.config import load_config
from tm_live_agent.controller import RuleController
from tm_live_agent.debug_view import show_debug
from tm_live_agent.hotkeys import HotkeyToggle
from tm_live_agent.tminterface_bridge import ControlCommand, TMInterfaceBridge
from tm_live_agent.vision import RoadVision
from tm_live_agent.window_capture import WindowCapture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runtime = cfg.get("runtime", {})
    wcfg = cfg.get("window", {})

    print("[agent] Starting TMInterface bridge...")
    bridge = TMInterfaceBridge(server_name=runtime.get("server_name", "TMInterface0"))
    bridge.start()
    if not bridge.wait_until_connected(timeout_s=10):
        raise RuntimeError("Could not connect to TMInterface. Launch TMNF through TMInterface 1.4.3 first.")

    print("[agent] Locating TMNF window...")
    capture = WindowCapture(
        title_substring=wcfg.get("title_substring", "TmForever"),
        crop=(
            float(wcfg.get("crop_left", 0.0)),
            float(wcfg.get("crop_top", 0.0)),
            float(wcfg.get("crop_right", 1.0)),
            float(wcfg.get("crop_bottom", 1.0)),
        ),
    )
    print(f"[agent] Capturing: {capture.window.title} {capture.window.width}x{capture.window.height}")

    hotkeys = HotkeyToggle(
        start_key=runtime.get("start_key", "F8"),
        stop_key=runtime.get("stop_key", "ESC"),
    )
    vision = RoadVision(cfg)
    controller = RuleController(cfg)

    control_hz = float(runtime.get("control_hz", 20))
    dt_target = 1.0 / max(1.0, control_hz)
    debug_enabled = bool(runtime.get("debug", True)) and not args.no_debug
    window_name = runtime.get("debug_window_name", "TMNF Live Agent Debug")

    active = False
    bridge.set_active(False)
    bridge.set_command(ControlCommand())
    print("[agent] Ready. Press F8 to start/pause. Press ESC for emergency stop.")
    print("[agent] Keep the debug window off the TMNF window so it does not cover the captured game image.")

    try:
        while True:
            loop_start = time.monotonic()

            if hotkeys.stop_pressed():
                print("[agent] ESC pressed: emergency stop.")
                active = False
                bridge.release()
                break

            if hotkeys.start_pressed():
                active = not active
                bridge.set_active(active)
                if not active:
                    bridge.set_command(ControlCommand())
                    controller.reset()
                print(f"[agent] {'ACTIVE' if active else 'PAUSED'}")

            frame = capture.grab()
            vis = vision.process(frame)
            telem = bridge.get_telemetry()
            cmd, ctrl_dbg = controller.compute(vis, telem, active=active)

            if active:
                bridge.set_active(True)
                bridge.set_command(cmd)
            else:
                bridge.set_command(ControlCommand())

            if debug_enabled:
                key = show_debug(window_name, vis, cmd, telem, ctrl_dbg)
                if key in (ord("q"), 27):
                    print("[agent] Debug window quit key pressed.")
                    active = False
                    bridge.release()
                    break

            elapsed = time.monotonic() - loop_start
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)
    finally:
        bridge.release()
        cv2.destroyAllWindows()
        print("[agent] Inputs released. Done.")


if __name__ == "__main__":
    main()
