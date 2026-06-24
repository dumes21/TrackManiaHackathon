from __future__ import annotations

from dataclasses import dataclass
from math import radians, sin, cos
from typing import Any, Dict

import cv2
import numpy as np


@dataclass
class RayHit:
    angle_deg: float
    distance_frac: float
    end_xy: tuple[int, int]


@dataclass
class VisionResult:
    frame_bgr: np.ndarray
    roi_bgr: np.ndarray
    road_mask: np.ndarray
    debug_bgr: np.ndarray
    road_center_error: float
    best_ray_error: float
    front_clearance: float
    confidence: float
    rays: list[RayHit]
    target_xy: tuple[int, int]


def _rel_box(shape: tuple[int, int], box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    l, t, r, b = box
    x1 = int(np.clip(l, 0, 1) * w)
    y1 = int(np.clip(t, 0, 1) * h)
    x2 = int(np.clip(r, 0, 1) * w)
    y2 = int(np.clip(b, 0, 1) * h)
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def _largest_component(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    keep = np.zeros_like(mask)
    best_label = -1
    best_area = 0

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area and area > best_area:
            best_area = area
            best_label = label_id

    if best_label != -1:
        keep[labels == best_label] = 255
    return keep


def _raycast(
    mask: np.ndarray,
    origin: tuple[int, int],
    angle_deg: float,
    length_px: int,
    step_px: int,
    fail_patience: int,
) -> RayHit:
    h, w = mask.shape[:2]
    ox, oy = origin
    theta = radians(angle_deg)
    dx = sin(theta)
    dy = -cos(theta)

    last_good = (ox, oy)
    consecutive_bad = 0
    travelled = 0

    for d in range(0, length_px, max(1, step_px)):
        x = int(round(ox + dx * d))
        y = int(round(oy + dy * d))

        if x < 0 or x >= w or y < 0 or y >= h:
            travelled = d
            break

        if mask[y, x] > 0:
            last_good = (x, y)
            travelled = d
            consecutive_bad = 0
        else:
            consecutive_bad += 1
            if consecutive_bad >= fail_patience:
                travelled = max(0, d - fail_patience * step_px)
                break
    else:
        travelled = length_px
        last_good = (
            int(round(ox + dx * length_px)),
            int(round(oy + dy * length_px)),
        )

    last_good = (
        int(np.clip(last_good[0], 0, w - 1)),
        int(np.clip(last_good[1], 0, h - 1)),
    )

    return RayHit(
        angle_deg=float(angle_deg),
        distance_frac=float(np.clip(travelled / max(1, length_px), 0, 1)),
        end_xy=last_good,
    )


class RoadVision:
    """Light-grey road detector for Trackmania tm_env."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def process(self, frame_bgr: np.ndarray) -> VisionResult:
        vcfg = self.cfg.get("vision", {})
        roi_box = (
            float(vcfg.get("roi_left", 0.00)),
            float(vcfg.get("roi_top", 0.18)),
            float(vcfg.get("roi_right", 1.00)),
            float(vcfg.get("roi_bottom", 0.92)),
        )

        x1, y1, x2, y2 = _rel_box(frame_bgr.shape, roi_box)
        roi = frame_bgr[y1:y2, x1:x2].copy()
        h, w = roi.shape[:2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        _, S, V = cv2.split(hsv)

        sat_max = int(vcfg.get("generic_max_saturation", 72))
        val_min = int(vcfg.get("generic_min_value", 38))
        val_max = int(vcfg.get("generic_max_value", 250))

        mask = ((S <= sat_max) & (V >= val_min) & (V <= val_max))
        road_mask = np.where(mask, 255, 0).astype(np.uint8)

        k = int(vcfg.get("morph_kernel", 7))
        k = max(1, k if k % 2 == 1 else k + 1)
        kernel = np.ones((k, k), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

        min_area = int(float(vcfg.get("min_component_area_ratio", 0.0025)) * h * w)
        road_mask = _largest_component(road_mask, min_area=min_area)

        confidence = float(np.count_nonzero(road_mask) / max(1, h * w))

        look_rows = [0.42, 0.54, 0.66, 0.78, 0.88]
        weights = [2.5, 1.5, 1.0, 0.5, 0.2]
        centers = []
        center_weights = []

        # --- NEW: Smart Road Clustering ---
        for rel_y, wt in zip(look_rows, weights):
            yy = int(rel_y * h)
            band = road_mask[max(0, yy - 4):min(h, yy + 5), :]
            col_sums = np.sum(band, axis=0)
            xs = np.where(col_sums > 0)[0]
            
            if len(xs) > max(10, 0.02 * w):
                # Find gaps in the road (e.g., the hole) that are wider than 8% of the screen
                gaps = np.diff(xs)
                gap_indices = np.where(gaps > 0.08 * w)[0]
                
                if len(gap_indices) > 0:
                    # The road is split! Divide xs into separate segments
                    segments = np.split(xs, gap_indices + 1)
                    # Pick the widest patch of light grey road
                    best_segment = max(segments, key=len)
                    centers.append(float(np.mean(best_segment)))
                else:
                    # Normal continuous road
                    centers.append(float(np.mean(xs)))
                center_weights.append(float(wt))

        road_center_x = float(np.average(centers, weights=center_weights)) if centers else w / 2
        origin = (w // 2, int(float(vcfg.get("ray_origin_y", 0.88)) * h))
        road_center_error = float(np.clip((road_center_x - origin[0]) / max(1, w / 2), -1.0, 1.0))

        ray_count = max(3, int(vcfg.get("ray_count", 20)))
        fov = float(vcfg.get("ray_fov_degrees", 200))
        max_len = int(float(vcfg.get("ray_length_ratio", 0.88)) * h)
        step = int(vcfg.get("ray_step_px", 4))
        fail_patience = int(vcfg.get("ray_fail_patience", 4))
        angles = np.linspace(-fov / 2.0, fov / 2.0, ray_count)
        rays = [_raycast(road_mask, origin, float(a), max_len, step, fail_patience) for a in angles]

        best = max(rays, key=lambda r: r.distance_frac)
        best_ray_error = float(np.clip(best.angle_deg / max(1.0, fov / 2.0), -1.0, 1.0))
        front = min(rays, key=lambda r: abs(r.angle_deg))
        front_clearance = float(front.distance_frac)

        target_x = int(np.clip(road_center_x, 0, w - 1))
        target_y = int(0.45 * h)

        debug = self.draw_debug(roi, road_mask, rays, origin, (target_x, target_y), road_center_error)

        return VisionResult(
            frame_bgr=frame_bgr,
            roi_bgr=roi,
            road_mask=road_mask,
            debug_bgr=debug,
            road_center_error=road_center_error,
            best_ray_error=best_ray_error,
            front_clearance=front_clearance,
            confidence=confidence,
            rays=rays,
            target_xy=(target_x, target_y),
        )

    @staticmethod
    def draw_debug(
        roi: np.ndarray,
        road_mask: np.ndarray,
        rays: list[RayHit],
        origin: tuple[int, int],
        target_xy: tuple[int, int],
        road_center_error: float = 0.0,
    ) -> np.ndarray:
        h, w = roi.shape[:2]
        overlay = roi.copy()

        mask_color = np.zeros_like(overlay)
        mask_color[:, :, 1] = road_mask
        overlay = cv2.addWeighted(overlay, 0.70, mask_color, 0.30, 0)

        for r in rays:
            if r.distance_frac > 0.55:
                color = (0, 220, 0)
            elif r.distance_frac > 0.32:
                color = (0, 180, 220)
            else:
                color = (0, 0, 255)
            cv2.line(overlay, origin, r.end_xy, color, 2)
            cv2.circle(overlay, r.end_xy, 3, color, -1)

        for yy in range(0, h, 20):
            cv2.line(overlay, (w // 2, yy), (w // 2, min(h, yy + 10)), (255, 255, 255), 1)

        road_cx = int(np.clip((road_center_error * (w / 2)) + (w / 2), 0, w - 1))
        cv2.line(overlay, (road_cx, 0), (road_cx, h), (255, 0, 255), 2)

        label_color = (0, 255, 0) if abs(road_center_error) < 0.12 else (0, 0, 255)
        cv2.putText(
            overlay,
            f"err:{road_center_error:+.2f}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            label_color,
            1,
        )

        cv2.circle(overlay, origin, 6, (255, 255, 255), -1)
        cv2.circle(overlay, target_xy, 7, (255, 0, 255), -1)
        cv2.line(overlay, origin, target_xy, (255, 0, 255), 2)

        return overlay