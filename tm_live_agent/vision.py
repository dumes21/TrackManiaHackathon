from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from typing import Any, Dict, List, Tuple

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


def _largest_seed_connected_component(mask: np.ndarray, seed_rect: tuple[int, int, int, int], min_area: int) -> np.ndarray:
    """Keep connected components that intersect the seed rectangle.

    If none intersects, fall back to the largest sufficiently large component.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    sx1, sy1, sx2, sy2 = seed_rect
    seed_labels = labels[sy1:sy2, sx1:sx2]
    label_ids, counts = np.unique(seed_labels[seed_labels > 0], return_counts=True)

    keep = np.zeros_like(mask)
    if len(label_ids) > 0:
        # Keep the seed-touching component with the most pixels in the seed patch.
        chosen = int(label_ids[np.argmax(counts)])
        if stats[chosen, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == chosen] = 255
            return keep

    # Fallback: largest non-background component.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    if stats[largest_idx, cv2.CC_STAT_AREA] >= min_area:
        keep[labels == largest_idx] = 255
    return keep


def _raycast(mask: np.ndarray, origin: tuple[int, int], angle_deg: float, length_px: int, step_px: int, fail_patience: int) -> RayHit:
    """Cast an image-space pseudo-LIDAR ray.

    Angle 0 points straight up the image. Negative is left, positive is right.
    """
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
        last_good = (int(round(ox + dx * length_px)), int(round(oy + dy * length_px)))
        last_good = (int(np.clip(last_good[0], 0, w - 1)), int(np.clip(last_good[1], 0, h - 1)))

    return RayHit(angle_deg=float(angle_deg), distance_frac=float(np.clip(travelled / max(1, length_px), 0, 1)), end_xy=last_good)


class RoadVision:
    """Open-path detector using adaptive road masking and pseudo-LIDAR rays.

    This is deliberately not a CNN. It is fast, explainable, and debuggable.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def process(self, frame_bgr: np.ndarray) -> VisionResult:
        vcfg = self.cfg.get("vision", {})
        roi_box = (
            float(vcfg.get("roi_left", 0.06)),
            float(vcfg.get("roi_top", 0.28)),
            float(vcfg.get("roi_right", 0.94)),
            float(vcfg.get("roi_bottom", 0.94)),
        )
        x1, y1, x2, y2 = _rel_box(frame_bgr.shape, roi_box)
        roi = frame_bgr[y1:y2, x1:x2].copy()
        h, w = roi.shape[:2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        sx_half = int(float(vcfg.get("seed_x_half_width", 0.12)) * w)
        sy1 = int(float(vcfg.get("seed_y_top", 0.64)) * h)
        sy2 = int(float(vcfg.get("seed_y_bottom", 0.88)) * h)
        sx1 = max(0, w // 2 - sx_half)
        sx2 = min(w, w // 2 + sx_half)
        seed_rect = (sx1, sy1, sx2, sy2)
        seed = hsv[sy1:sy2, sx1:sx2]

        # Estimate local road appearance from seed patch. Ignore extremely dark pixels.
        seed_flat = seed.reshape(-1, 3)
        seed_flat = seed_flat[seed_flat[:, 2] > 25]
        if len(seed_flat) < 20:
            med_h, med_s, med_v = 0, 40, 100
        else:
            med_h, med_s, med_v = np.median(seed_flat, axis=0)

        sat_tol = float(vcfg.get("adaptive_sat_tol", 55))
        val_tol = float(vcfg.get("adaptive_val_tol", 75))
        adaptive = (np.abs(S.astype(np.float32) - med_s) <= sat_tol) & (np.abs(V.astype(np.float32) - med_v) <= val_tol)

        # For grey road surfaces, hue is unstable, so use a generic low-saturation prior.
        generic = (
            (S <= int(vcfg.get("generic_max_saturation", 95)))
            & (V >= int(vcfg.get("generic_min_value", 35)))
            & (V <= int(vcfg.get("generic_max_value", 245)))
        )

        mask = np.where(adaptive | generic, 255, 0).astype(np.uint8)

        k = int(vcfg.get("morph_kernel", 5))
        k = max(1, k if k % 2 == 1 else k + 1)
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        min_area = int(float(vcfg.get("min_component_area_ratio", 0.003)) * h * w)
        road_mask = _largest_seed_connected_component(mask, seed_rect, min_area=min_area)

        # Confidence: fraction of ROI that is connected to the local road seed.
        confidence = float(np.count_nonzero(road_mask) / max(1, h * w))

        # Road centre from lookahead bands. Prefer rows farther ahead but not too near horizon.
        look_rows = [0.38, 0.48, 0.58, 0.68]
        weights = [1.35, 1.15, 0.95, 0.75]
        centers: list[float] = []
        center_weights: list[float] = []
        for rel_y, wt in zip(look_rows, weights):
            yy = int(rel_y * h)
            band = road_mask[max(0, yy - 4):min(h, yy + 5), :]
            xs = np.where(band > 0)[1]
            if len(xs) > max(10, 0.02 * w):
                centers.append(float(np.mean(xs)))
                center_weights.append(float(wt))
        if centers:
            road_center_x = float(np.average(centers, weights=center_weights))
        else:
            road_center_x = w / 2
        origin = (w // 2, int(0.90 * h))
        road_center_error = float(np.clip((road_center_x - origin[0]) / max(1, w / 2), -1.0, 1.0))

        # Pseudo-LIDAR rays.
        ray_count = int(vcfg.get("ray_count", 13))
        ray_count = max(3, ray_count)
        fov = float(vcfg.get("ray_fov_degrees", 110))
        max_len = int(float(vcfg.get("ray_length_ratio", 0.80)) * h)
        step = int(vcfg.get("ray_step_px", 4))
        fail_patience = int(vcfg.get("ray_fail_patience", 3))
        angles = np.linspace(-fov / 2.0, fov / 2.0, ray_count)
        rays = [_raycast(road_mask, origin, float(a), max_len, step, fail_patience) for a in angles]
        best = max(rays, key=lambda r: r.distance_frac)
        best_ray_error = float(np.clip(best.angle_deg / max(1.0, fov / 2.0), -1.0, 1.0))
        front = min(rays, key=lambda r: abs(r.angle_deg))
        front_clearance = float(front.distance_frac)

        target_x = int(np.clip(road_center_x, 0, w - 1))
        target_y = int(0.45 * h)

        debug = self.draw_debug(roi, road_mask, rays, origin, (target_x, target_y), seed_rect)
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
        seed_rect: tuple[int, int, int, int],
    ) -> np.ndarray:
        overlay = roi.copy()
        mask_color = np.zeros_like(overlay)
        mask_color[:, :, 1] = road_mask
        overlay = cv2.addWeighted(overlay, 0.70, mask_color, 0.30, 0)

        for r in rays:
            # Shorter rays are drawn darker red; longer rays greener.
            if r.distance_frac > 0.55:
                color = (0, 220, 0)
            elif r.distance_frac > 0.32:
                color = (0, 180, 220)
            else:
                color = (0, 0, 255)
            cv2.line(overlay, origin, r.end_xy, color, 2)
            cv2.circle(overlay, r.end_xy, 3, color, -1)

        cv2.circle(overlay, origin, 6, (255, 255, 255), -1)
        cv2.circle(overlay, target_xy, 7, (255, 0, 255), -1)
        cv2.line(overlay, origin, target_xy, (255, 0, 255), 2)
        sx1, sy1, sx2, sy2 = seed_rect
        cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (255, 255, 0), 2)
        return overlay
