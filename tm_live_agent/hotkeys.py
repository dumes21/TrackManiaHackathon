from __future__ import annotations

import time

try:
    import win32api
except ImportError:  # pragma: no cover - Windows-only runtime dependency
    win32api = None


VK_CODES = {
    "ESC": 0x1B,
    "F8": 0x77,
    "F9": 0x78,
    "SPACE": 0x20,
}


class HotkeyToggle:
    """Simple Windows hotkey edge detector using GetAsyncKeyState.

    This avoids global keyboard hooks and usually works without admin rights.
    """

    def __init__(self, start_key: str = "F8", stop_key: str = "ESC", debounce_s: float = 0.25):
        if win32api is None:
            raise RuntimeError("pywin32 is required for hotkeys on Windows. Run: pip install pywin32")
        self.start_vk = VK_CODES.get(start_key.upper())
        self.stop_vk = VK_CODES.get(stop_key.upper())
        if self.start_vk is None:
            raise ValueError(f"Unsupported start key: {start_key}")
        if self.stop_vk is None:
            raise ValueError(f"Unsupported stop key: {stop_key}")
        self.debounce_s = debounce_s
        self._last_start = 0.0
        self._last_stop = 0.0

    @staticmethod
    def _down(vk: int) -> bool:
        # High bit means currently pressed.
        return bool(win32api.GetAsyncKeyState(vk) & 0x8000)

    def start_pressed(self) -> bool:
        now = time.monotonic()
        if self._down(self.start_vk) and now - self._last_start > self.debounce_s:
            self._last_start = now
            return True
        return False

    def stop_pressed(self) -> bool:
        now = time.monotonic()
        if self._down(self.stop_vk) and now - self._last_stop > self.debounce_s:
            self._last_stop = now
            return True
        return False
