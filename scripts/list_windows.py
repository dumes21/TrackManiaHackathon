from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tm_live_agent.window_capture import list_visible_windows


if __name__ == "__main__":
    for hwnd, title in list_visible_windows():
        print(f"{hwnd}: {title}")
