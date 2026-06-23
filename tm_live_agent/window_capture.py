from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mss
import numpy as np

try:
    import win32gui
except ImportError:  # pragma: no cover - Windows-only runtime dependency
    win32gui = None


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int


def list_visible_windows() -> list[tuple[int, str]]:
    """Return visible Windows window handles and titles."""
    if win32gui is None:
        raise RuntimeError("pywin32 is required on Windows. Run: pip install pywin32")
    windows: list[tuple[int, str]] = []

    def enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title.strip():
                windows.append((hwnd, title))

    win32gui.EnumWindows(enum_handler, None)
    return windows


def find_window(title_substring: str) -> WindowInfo:
    """Find a visible window whose title contains title_substring.

    Returns the client-area rectangle in screen coordinates, which avoids title bars
    and borders. This is important for stable vision crops.
    """
    if win32gui is None:
        raise RuntimeError("pywin32 is required on Windows. Run: pip install pywin32")

    title_substring = title_substring.lower()
    match: Optional[tuple[int, str]] = None
    for hwnd, title in list_visible_windows():
        if title_substring in title.lower():
            match = (hwnd, title)
            break
    if match is None:
        titles = "\n".join(f"- {title}" for _, title in list_visible_windows()[:30])
        raise RuntimeError(
            f"Could not find a visible window containing '{title_substring}'.\n"
            f"Open TMNF windowed through TMInterface first. Visible windows:\n{titles}"
        )

    hwnd, title = match
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    screen_left, screen_top = win32gui.ClientToScreen(hwnd, (left, top))
    screen_right, screen_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
    return WindowInfo(
        hwnd=hwnd,
        title=title,
        left=screen_left,
        top=screen_top,
        width=max(1, screen_right - screen_left),
        height=max(1, screen_bottom - screen_top),
    )


class WindowCapture:
    """Fast-ish screen capture of the TMNF client area using mss."""

    def __init__(
        self,
        title_substring: str = "TmForever",
        crop: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    ):
        self.title_substring = title_substring
        self.crop = crop
        self.sct = mss.mss()
        self.window = find_window(title_substring)

    def refresh_window(self) -> WindowInfo:
        self.window = find_window(self.title_substring)
        return self.window

    def get_bbox(self) -> dict[str, int]:
        w = self.window
        l_rel, t_rel, r_rel, b_rel = self.crop
        left = int(w.left + l_rel * w.width)
        top = int(w.top + t_rel * w.height)
        width = int((r_rel - l_rel) * w.width)
        height = int((b_rel - t_rel) * w.height)
        return {"left": left, "top": top, "width": max(1, width), "height": max(1, height)}

    def grab(self) -> np.ndarray:
        """Capture current window client area as BGR image."""
        bbox = self.get_bbox()
        img = np.array(self.sct.grab(bbox), dtype=np.uint8)
        # mss gives BGRA; OpenCV usually expects BGR.
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
