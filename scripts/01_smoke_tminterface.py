"""Smoke test for Python <-> TMInterface 1.4.3.

Run TMNF through TMInterface, load a map, sit on the start line, then:
    python scripts/01_smoke_tminterface.py

The car should accelerate straight and telemetry should print.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

from tm_live_agent.tminterface_bridge import ControlCommand, TMInterfaceBridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="TMInterface0")
    args = parser.parse_args()

    bridge = TMInterfaceBridge(server_name=args.server)
    bridge.start()
    if not bridge.wait_until_connected(timeout_s=10):
        raise RuntimeError("Could not connect to TMInterface. Is TMNF running through TMInterface 1.4.3?")

    print("[smoke] Connected. Holding accelerate for 8 seconds. Ctrl+C to stop.")
    bridge.set_active(True)
    bridge.set_command(ControlCommand(steer=0.0, accelerate=True, brake=False))
    try:
        start = time.monotonic()
        while time.monotonic() - start < 8.0:
            t = bridge.get_telemetry()
            print(
                f"t={t.race_time_ms/1000:6.2f}s  speed={t.speed_kmh:6.1f} km/h  "
                f"pos=({t.position[0]:7.1f},{t.position[1]:6.1f},{t.position[2]:7.1f})  "
                f"yaw={t.yaw_pitch_roll[0]:+.2f}"
            )
            time.sleep(0.5)
    finally:
        bridge.release()
        print("[smoke] Released inputs.")


if __name__ == "__main__":
    main()
