"""Check TMNF window capture and vision debug output.

Run TMNF windowed first, then:
    python scripts/02_check_capture.py --config config/default.yaml

Press q or ESC in the debug window to quit.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import cv2

from tm_live_agent.config import load_config
from tm_live_agent.window_capture import WindowCapture
from tm_live_agent.vision import RoadVision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    wcfg = cfg.get("window", {})
    cap = WindowCapture(
        title_substring=wcfg.get("title_substring", "TmForever"),
        crop=(
            float(wcfg.get("crop_left", 0.0)),
            float(wcfg.get("crop_top", 0.0)),
            float(wcfg.get("crop_right", 1.0)),
            float(wcfg.get("crop_bottom", 1.0)),
        ),
    )
    print(f"[capture] Found window: {cap.window.title} {cap.window.width}x{cap.window.height}")
    vision = RoadVision(cfg)

    while True:
        frame = cap.grab()
        result = vision.process(frame)
        view = result.debug_bgr
        if view.shape[1] > 900:
            scale = 900 / view.shape[1]
            view = cv2.resize(view, (900, int(view.shape[0] * scale)))
        cv2.imshow("TMNF Capture + Vision Check", view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
