from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from tm_live_agent.tminterface_bridge import ControlCommand, Telemetry
from tm_live_agent.vision import VisionResult


@dataclass
class ControlDebug:
    active: bool
    mode: str
    desired_steer: float
    target_speed_kmh: float
    speed_kmh: float
    road_center_error: float
    best_ray_error: float
    front_clearance: float
    confidence: float
    stuck_seconds: float


class RuleController:
    """Conservative road-following controller for finishing, not speedrunning."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("controller", {})
        self.prev_error = 0.0
        self.prev_time = time.monotonic()
        self.low_speed_since: float | None = None
        self.recovery_start: float | None = None

    def reset(self) -> None:
        self.prev_error = 0.0
        self.prev_time = time.monotonic()
        self.low_speed_since = None
        self.recovery_start = None

    def compute(self, vision: VisionResult, telemetry: Telemetry, active: bool) -> tuple[ControlCommand, ControlDebug]:
        now = time.monotonic()
        dt = max(1e-3, now - self.prev_time)
        self.prev_time = now

        speed = telemetry.speed_kmh
        min_conf = float(self.cfg.get("min_confidence", 0.035))
        low_front = float(self.cfg.get("low_front_clearance", 0.32))
        med_front = float(self.cfg.get("medium_front_clearance", 0.48))

        center_bias = float(self.cfg.get("center_bias", 0.45))
        ray_bias = float(self.cfg.get("ray_bias", 0.55))
        blended_error = center_bias * vision.road_center_error + ray_bias * vision.best_ray_error

        # Steering PID-ish control.
        kp = float(self.cfg.get("steer_kp", 1.15))
        kd = float(self.cfg.get("steer_kd", 0.30))
        deriv = (blended_error - self.prev_error) / dt
        self.prev_error = blended_error
        steer = kp * blended_error + kd * deriv
        steer = float(np.clip(steer, -float(self.cfg.get("max_steer", 1.0)), float(self.cfg.get("max_steer", 1.0))))
        if abs(steer) < float(self.cfg.get("steer_deadzone", 0.03)):
            steer = 0.0

        # Stuck detection.
        stuck_speed = float(self.cfg.get("stuck_speed_kmh", 5))
        stuck_after = float(self.cfg.get("stuck_after_seconds", 1.6))
        if active and speed < stuck_speed:
            if self.low_speed_since is None:
                self.low_speed_since = now
        else:
            self.low_speed_since = None
            self.recovery_start = None
        stuck_seconds = 0.0 if self.low_speed_since is None else now - self.low_speed_since
        recovering = bool(self.cfg.get("recovery_enabled", True)) and stuck_seconds >= stuck_after

        mode = "paused"
        target_speed = 0.0
        cmd = ControlCommand()

        if not active:
            self.reset()
            return cmd, ControlDebug(
                active=False,
                mode=mode,
                desired_steer=0.0,
                target_speed_kmh=0.0,
                speed_kmh=speed,
                road_center_error=vision.road_center_error,
                best_ray_error=vision.best_ray_error,
                front_clearance=vision.front_clearance,
                confidence=vision.confidence,
                stuck_seconds=0.0,
            )

        if recovering:
            if self.recovery_start is None:
                self.recovery_start = now
            mode = "recovery"
            # Reverse/brake and steer toward the best visible opening. If uncertain,
            # alternate steering so the car can wiggle free. No auto-respawn.
            alt_period = float(self.cfg.get("recovery_alternate_seconds", 1.2))
            alt = -1.0 if int((now - self.recovery_start) / alt_period) % 2 else 1.0
            open_dir = np.sign(vision.best_ray_error) if abs(vision.best_ray_error) > 0.15 else alt
            rec_steer = float(self.cfg.get("recovery_steer", 0.75)) * float(open_dir)
            cmd = ControlCommand(steer=rec_steer, accelerate=False, brake=True)
            return cmd, ControlDebug(
                active=True,
                mode=mode,
                desired_steer=rec_steer,
                target_speed_kmh=0.0,
                speed_kmh=speed,
                road_center_error=vision.road_center_error,
                best_ray_error=vision.best_ray_error,
                front_clearance=vision.front_clearance,
                confidence=vision.confidence,
                stuck_seconds=stuck_seconds,
            )

        # Target speed is conservative and depends on confidence, clearance, and turn demand.
        max_speed = float(self.cfg.get("max_speed_kmh", 105))
        corner_speed = float(self.cfg.get("corner_speed_kmh", 65))
        unsure_speed = float(self.cfg.get("unsure_speed_kmh", 35))
        sharp_abs = float(self.cfg.get("sharp_turn_abs_steer", 0.52))

        target_speed = max_speed
        mode = "drive"
        if vision.confidence < min_conf:
            target_speed = unsure_speed
            mode = "low_confidence"
        elif vision.front_clearance < low_front:
            target_speed = min(unsure_speed, 35)
            mode = "blocked_front"
        elif vision.front_clearance < med_front:
            target_speed = min(corner_speed, 55)
            mode = "short_clearance"
        elif abs(steer) > sharp_abs:
            target_speed = corner_speed
            mode = "corner"

        # Throttle/brake logic. Brake only when meaningfully above target or path unsafe.
        accel = speed < target_speed and vision.confidence >= min_conf * 0.75 and vision.front_clearance >= low_front * 0.75
        brake = False
        if speed > target_speed + 12:
            brake = True
            accel = False
        if vision.confidence < min_conf * 0.6 or vision.front_clearance < low_front * 0.65:
            brake = speed > 18
            accel = False

        cmd = ControlCommand(steer=steer, accelerate=bool(accel), brake=bool(brake))
        return cmd, ControlDebug(
            active=True,
            mode=mode,
            desired_steer=steer,
            target_speed_kmh=target_speed,
            speed_kmh=speed,
            road_center_error=vision.road_center_error,
            best_ray_error=vision.best_ray_error,
            front_clearance=vision.front_clearance,
            confidence=vision.confidence,
            stuck_seconds=stuck_seconds,
        )
