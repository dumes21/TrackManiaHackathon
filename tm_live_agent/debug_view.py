from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from tm_live_agent.controller import ControlDebug
from tm_live_agent.tminterface_bridge import ControlCommand, Telemetry
from tm_live_agent.vision import VisionResult


def draw_status_panel(debug_img: np.ndarray, vision: VisionResult, cmd: ControlCommand, telem: Telemetry, ctrl: ControlDebug) -> np.ndarray:
    img = debug_img.copy()
    h, w = img.shape[:2]
    panel_h = 160
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)

    status = "ACTIVE" if ctrl.active else "PAUSED"
    lines = [
        f"F8 start/pause | ESC emergency stop | status={status} mode={ctrl.mode}",
        f"speed={telem.speed_kmh:5.1f} km/h  target={ctrl.target_speed_kmh:5.1f}  cp={telem.checkpoint_current}/{telem.checkpoint_target}",
        f"steer={cmd.steer:+.2f}  accel={int(cmd.accelerate)}  brake/rev={int(cmd.brake)}",
        f"center_err={vision.road_center_error:+.2f}  best_ray={vision.best_ray_error:+.2f}  front={vision.front_clearance:.2f}",
        f"confidence={vision.confidence:.3f}  stuck_s={ctrl.stuck_seconds:.1f}",
    ]
    y = 26
    for i, line in enumerate(lines):
        color = (0, 255, 0) if i == 0 and ctrl.active else (220, 220, 220)
        if i == 0 and not ctrl.active:
            color = (0, 220, 255)
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        y += 28

    return np.vstack([img, panel])


def show_debug(window_name: str, vision: VisionResult, cmd: ControlCommand, telem: Telemetry, ctrl: ControlDebug) -> int:
    view = draw_status_panel(vision.debug_bgr, vision, cmd, telem, ctrl)
    # Fit debug window to a practical size.
    max_w = 900
    if view.shape[1] > max_w:
        scale = max_w / view.shape[1]
        view = cv2.resize(view, (max_w, int(view.shape[0] * scale)))
    cv2.imshow(window_name, view)
    return cv2.waitKey(1) & 0xFF
