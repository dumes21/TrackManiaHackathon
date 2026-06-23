"""Drive and monitor a TMInterface bruteforce session for the seen course.

This does NOT replace TMInterface's built-in bruteforce optimizer. It registers a
client so the bruteforce script runs, defers each step to the built-in evaluation
(which optimizes for the configured target, e.g. finish time), and whenever the
optimizer accepts a faster solution it auto-saves the inputs to a text file in
TMInterface script syntax so you have a replayable artifact.

Workflow (full steps in docs/TAS_PLAN.md):
    1. Launch TMNF through TMInterface 1.4.3 and load the seen course.
    2. In the TMInterface console, load the baseline inputs and start bruteforce:
           load tas/bytesandbolts.txt
           set bf_search_forever true
           bruteforce
    3. Run this script. It connects, watches the search, and writes the best
       inputs to --output each time the optimizer improves.

    python scripts/tas_bruteforce.py --output tas/course1_optimized.txt

Press Ctrl+C to stop monitoring (the bruteforce keeps running in-game).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import threading
import time

from tminterface.client import Client
from tminterface.interface import TMInterface
from tminterface.structs import BFPhase


class BruteforceMonitor(Client):
    """Defers to TMInterface's built-in optimizer and saves improved runs."""

    def __init__(self, output_path: Path):
        super().__init__()
        self.output_path = output_path
        self._last_phase: int | None = None
        self._initial_max_time = 0
        self._best_time: int | None = None
        self._lock = threading.RLock()

    def on_registered(self, iface: TMInterface):
        print(f"[tas] Registered to server: {iface.server_name}")
        print("[tas] Watching bruteforce. Start it in the TMInterface console if you have not yet.")
        print(f"[tas] Improved inputs will be written to: {self.output_path}")

    def on_deregistered(self, iface: TMInterface):
        print("[tas] Deregistered from server.")

    def on_bruteforce_evaluate(self, iface: TMInterface, info):
        # The INITIAL phase replays the current best solution from the start; the
        # SEARCH phase is where the optimizer mutates inputs. A SEARCH -> INITIAL
        # transition means a new best was just accepted, so we save it then.
        phase = int(info.phase)

        if phase == BFPhase.INITIAL:
            self._initial_max_time = max(self._initial_max_time, int(info.time))
            improvement_accepted = self._last_phase == BFPhase.SEARCH
            if improvement_accepted or self._best_time is None:
                self._save_best(iface, info)
        elif phase == BFPhase.SEARCH:
            # Leaving an INITIAL replay; remember the finish time we just observed.
            if self._last_phase == BFPhase.INITIAL and self._initial_max_time > 0:
                self._best_time = self._initial_max_time
            self._initial_max_time = 0

        self._last_phase = phase
        # Return None: let TMInterface run its built-in evaluation for the target.
        return None

    def _save_best(self, iface: TMInterface, info) -> None:
        try:
            buf = iface.get_event_buffer()
            commands = buf.to_commands_str()
        except Exception as exc:  # pragma: no cover - defensive, hackathon runtime
            print(f"[tas] Could not read/serialize event buffer: {exc}")
            return

        with self._lock:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(commands, encoding="utf-8")

        time_ms = self._best_time if self._best_time is not None else int(info.time)
        if time_ms and time_ms > 0:
            print(f"[tas] New best ~{time_ms / 1000:.3f}s -> saved inputs to {self.output_path}")
        else:
            print(f"[tas] Saved current inputs to {self.output_path}")


def run_client(client: Client, server_name: str) -> None:
    """Connect and run the client loop without installing signal handlers.

    Mirrors tminterface.client.run_client but omits signal.signal, which only
    works on the main thread (see tm_live_agent/tminterface_bridge.py).
    """
    iface = TMInterface(server_name)
    iface.register(client)
    while iface.running:
        time.sleep(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="TMInterface0")
    parser.add_argument(
        "--output",
        default="tas/course1_optimized.txt",
        help="Where to write improved inputs (TMInterface script syntax).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    client = BruteforceMonitor(output_path)

    thread = threading.Thread(target=run_client, args=(client, args.server), daemon=True)
    thread.start()

    print("[tas] Connecting to TMInterface... Ctrl+C to stop monitoring.")
    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[tas] Stopped monitoring. Bruteforce continues in-game until you stop it there.")


if __name__ == "__main__":
    main()
