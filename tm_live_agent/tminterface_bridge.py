from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from tminterface.interface import TMInterface
from tminterface.client import Client


ANALOG_MAX = 65536


@dataclass
class ControlCommand:
    steer: float = 0.0       # -1 full left, +1 full right
    accelerate: bool = False
    brake: bool = False      # also reverses when stopped in TMNF


@dataclass
class Telemetry:
    connected: bool = False
    race_time_ms: int = 0
    speed_kmh: float = 0.0
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_pitch_roll: tuple[float, float, float] = (0.0, 0.0, 0.0)
    checkpoint_current: int = 0
    checkpoint_target: int = 0
    finished: bool = False


class _LiveClient(Client):
    def __init__(self, bridge: "TMInterfaceBridge"):
        super().__init__()
        self.bridge = bridge

    def on_registered(self, iface: TMInterface):
        with self.bridge._lock:
            self.bridge._iface = iface
            self.bridge._connected = True
        print(f"[tm] Registered to server: {iface.server_name}")

    def on_run_step(self, iface: TMInterface, _time: int):
        # Keep inputs neutral during countdown. Telemetry is still useful.
        state = iface.get_simulation_state()
        with self.bridge._lock:
            self.bridge._race_time_ms = _time
            self.bridge._latest_state = state
            cmd = self.bridge._command
            active = self.bridge._active and _time >= 0

        if active:
            steer = int(float(np.clip(cmd.steer, -1.0, 1.0)) * ANALOG_MAX)
            iface.set_input_state(
                steer=steer,
                accelerate=bool(cmd.accelerate),
                brake=bool(cmd.brake),
            )
        else:
            iface.set_input_state(steer=0, accelerate=False, brake=False)

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        with self.bridge._lock:
            self.bridge._checkpoint_current = current
            self.bridge._checkpoint_target = target
            if target > 0 and current == target:
                self.bridge._finished = True
        if target > 0 and current == target:
            print("[tm] FINISH detected")
            iface.prevent_simulation_finish()
        else:
            print(f"[tm] Checkpoint {current}/{target}")


class TMInterfaceBridge:
    """Small live-control bridge around TMInterface 1.4.3 Python client.

    TMInterface owns the physics callback. Our main Python loop only updates the
    latest desired command. The callback applies that command every physics tick.
    """

    def __init__(self, server_name: str = "TMInterface0"):
        self.server_name = server_name
        self._lock = threading.RLock()
        self._iface: Optional[TMInterface] = None
        self._connected = False
        self._active = False
        self._command = ControlCommand()
        self._latest_state = None
        self._race_time_ms = 0
        self._checkpoint_current = 0
        self._checkpoint_target = 0
        self._finished = False
        self._client = _LiveClient(self)
        self._thread = threading.Thread(
            target=self._run_client,
            daemon=True,
        )

    def _run_client(self) -> None:
        """Run the TMInterface client loop.

        This mirrors ``tminterface.client.run_client`` but omits the
        ``signal.signal`` calls, which can only run on the main thread and
        would otherwise crash this background thread.
        """
        iface = TMInterface(self.server_name)
        iface.register(self._client)
        while iface.running:
            time.sleep(0)

    def start(self) -> None:
        self._thread.start()

    def wait_until_connected(self, timeout_s: float = 10.0) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            with self._lock:
                if self._connected:
                    return True
            time.sleep(0.05)
        return False

    def set_active(self, active: bool) -> None:
        with self._lock:
            self._active = active
            if not active:
                self._command = ControlCommand()

    def set_command(self, command: ControlCommand) -> None:
        with self._lock:
            self._command = ControlCommand(
                steer=float(np.clip(command.steer, -1.0, 1.0)),
                accelerate=bool(command.accelerate),
                brake=bool(command.brake),
            )

    def release(self) -> None:
        self.set_command(ControlCommand())
        self.set_active(False)

    def get_telemetry(self) -> Telemetry:
        with self._lock:
            s = self._latest_state
            if s is None:
                return Telemetry(connected=self._connected)
            try:
                pos = tuple(float(x) for x in s.position)
                vel = tuple(float(x) for x in s.velocity)
                ypr = tuple(float(x) for x in s.yaw_pitch_roll)
                speed = float(s.display_speed)
            except Exception:
                pos = (0.0, 0.0, 0.0)
                vel = (0.0, 0.0, 0.0)
                ypr = (0.0, 0.0, 0.0)
                speed = 0.0
            return Telemetry(
                connected=self._connected,
                race_time_ms=int(self._race_time_ms),
                speed_kmh=speed,
                position=pos,
                velocity=vel,
                yaw_pitch_roll=ypr,
                checkpoint_current=int(self._checkpoint_current),
                checkpoint_target=int(self._checkpoint_target),
                finished=bool(self._finished),
            )
