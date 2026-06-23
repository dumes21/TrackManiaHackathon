# TrackMania Nations Forever — Autonomous Racing & TAS Toolkit

Software for racing **TrackMania Nations Forever (TMNF)** as fast as possible with
as little human input as possible. The project tackles two kinds of track:

- **Known ("seen") course** — available ahead of time. The fastest, most reliable
  result here is a **Tool-Assisted run (TAS)**: a fixed, optimized input sequence
  replayed deterministically.
- **Unknown ("unseen") course** — only available briefly at test time. Here the
  project uses a **real-time autonomous agent** that watches the screen and drives
  on its own, with no prior knowledge of the track.

Everything is built on **TMInterface 1.4.3**, which gives clean programmatic input
injection and real telemetry (speed, position, checkpoints, finish) for TMNF.

---

## Solution approach

The project combines three techniques, each suited to a different situation:

| Technique | Best for | What it is |
|-----------|----------|------------|
| **TAS / bruteforce** | the known course | TMInterface's built-in optimizer mutates a finishing run's inputs to drive the lap time down. Deterministic, no human input. |
| **Rule-based agent** | the unknown course (reliable) | A conservative road-follower: detect the road, steer toward the open path, slow for corners, recover if stuck. |
| **Genetic-algorithm agent** | the unknown course (learned) | Neuroevolution: a small steering policy is evolved by trial-and-error against a reward. Throttle/brake are handled by the rule-based speed logic (a hybrid). |

All three autonomous pieces share one **vision system**: the screen is captured and
processed with OpenCV into a road/open-path mask, then **pseudo-LIDAR rays** are
cast through that mask to measure how far the drivable road extends in each
direction. That ray vector is what the controllers and the learned policy "see".

### How the autonomous agent perceives and drives

```text
capture TMNF window
    ↓
crop the road-ahead region (ROI)
    ↓
build a road / open-path mask (OpenCV)
    ↓
cast pseudo-LIDAR rays into the mask  ──►  ray distances + clearance + confidence
    ↓
decide steering          (rule-based PID, OR an evolved policy)
decide throttle / brake  (rule-based, speed- and corner-aware, using real telemetry)
    ↓
send controls through TMInterface
```

---

## Requirements

Use this setup for consistency:

```text
Windows
TrackMania Nations Forever
TMInterface 1.4.3
TMNF windowed at 1280x960
Windows taskbar auto-hidden
Normal behind-car camera
```

> Do **not** launch TMNF normally — always launch it through **TMInterface 1.4.3**,
> or the programmatic input/telemetry will not be available.

### Install

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Project structure

```text
config/default.yaml                 All tunable settings (window, vision, controller, evolution)

scripts/
  01_smoke_tminterface.py           Sanity check: Python <-> TMInterface link works
  02_check_capture.py               Sanity check: screen capture + vision overlay
  run_live_agent.py                 Rule-based autonomous agent (F8 start/stop)
  train_evolution.py                Genetic-algorithm trainer (neuroevolution)
  tas_bruteforce.py                 Monitors/saves a TMInterface bruteforce session
  list_windows.py                   Lists visible windows if capture can't find TMNF

tm_live_agent/                      Shared library
  tminterface_bridge.py             Input injection + telemetry + respawn
  window_capture.py                 TMNF window capture (mss + pywin32)
  vision.py                         Road mask + pseudo-LIDAR rays
  controller.py                     Speed-aware rule-based controller
  debug_view.py                     On-screen debug overlay
  hotkeys.py                        F8 / Esc handling
  config.py                         YAML config loader

Agent.py                            Genetic-algorithm policy + training loop
tm_env.py                           Gymnasium environment (TMInterface + vision backed)

tas/                                TAS input scripts (baseline + optimized)
docs/
  HACKATHON_RUNBOOK.md              Test-day checklist
  TAS_PLAN.md                       Known-course TAS workflow
  TROUBLESHOOTING.md                Common fixes
  DESIGN_NOTES.md                   Design rationale
```

---

## Usage

Run the sanity checks first; they catch most setup problems before you rely on
anything more complex. In all cases: launch TMNF through TMInterface, load a map,
and put the car on the start line.

### 1. Verify the TMInterface link

```powershell
python scripts/01_smoke_tminterface.py
```

The car should accelerate straight and telemetry should print. It may crash — that
is fine; this only proves the input/telemetry bridge works.

### 2. Verify capture + vision

```powershell
python scripts/02_check_capture.py --config config/default.yaml
```

A debug window should show the detected road mask (green), the pseudo-LIDAR rays,
a magenta target direction, and a cyan sample box near the lower centre. Press
`q` or `Esc` to quit.

> **Keep any debug window off the TMNF window.** If it overlaps the game, the
> screen capture will photograph the debug window instead of the road.

### 3a. Run the rule-based agent (unknown course, reliable)

```powershell
python scripts/run_live_agent.py --config config/default.yaml
```

Controls:

```text
F8  = start / pause autonomous control
Esc = emergency stop and release inputs
q   = quit (when the debug window is focused)
```

It does not auto-respawn — respawn manually if it gets badly stuck.

### 3b. Train the genetic-algorithm agent (unknown course, learned)

```powershell
# Train (no window, faster)
python scripts/train_evolution.py --config config/default.yaml

# Train while watching what the agent sees
python scripts/train_evolution.py --config config/default.yaml --debug
```

How training works each generation:

1. The car is respawned to the start line.
2. It drives one episode — the **evolved policy steers**, while the **rule-based
   logic handles throttle/brake** using real telemetry (so it brakes for corners).
3. The run is scored (reward = speed + staying off the walls + clear road ahead +
   checkpoint/finish bonuses − a crash penalty).
4. If the mutated policy beats the record, its weights are saved.

Outputs (in the directory you run from, configurable with `--weights`):

```text
best_trackmania_weights.npy         Current best policy (overwritten on improvement)
best_trackmania_weights_log.csv     Append-only history: timestamp, generation, best_reward
```

Press `Ctrl+C` to stop — the best weights are already saved. Re-running **resumes**
from the saved weights, so training accumulates across sessions.

> Note: this is **real-time** neuroevolution (one candidate per ~25 s episode), so
> improvement is gradual. Leave it running for long stretches to see progress.

### 4. Optimize a known course (TAS)

The fastest known-course result comes from TMInterface's built-in **bruteforce**,
seeded by any run that finishes the lap. See **`docs/TAS_PLAN.md`** for the full
workflow. Optionally, monitor and auto-save improvements with:

```powershell
python scripts/tas_bruteforce.py --output tas/course1_optimized.txt
```

---

## Configuration & tuning

All settings live in `config/default.yaml`, grouped by area:

- **`window`** — which window to capture and how to crop it.
- **`vision`** — region of interest, road-mask thresholds, and ray settings
  (`ray_count` must match the policy input size, currently **20**).
- **`controller`** — steering gains and speed targets used by both the rule-based
  agent and the throttle/brake half of the hybrid trainer.
- **`evolution`** — reward weights and episode termination for the GA trainer.

Common adjustments:

```yaml
controller:
  max_speed_kmh: 85       # lower = safer, higher = faster/riskier
  corner_speed_kmh: 50
  steer_kp: 1.35          # raise if it steers too weakly
  steer_kd: 0.15          # lower if it wiggles
  low_front_clearance: 0.25   # lower if it wrongly thinks the road is blocked
```

If capture can't find the game window, list windows and update the title:

```powershell
python scripts/list_windows.py
```
```yaml
window:
  title_substring: "whatever title you see"
```

---

## Known limitations

- The autonomous agent reads only the **visible** road, so it can struggle with
  loops, wallrides, jumps where the road disappears, unusual surfaces/lighting,
  and tracks where the visible road doesn't indicate the correct route.
- Real-time evolution is **slow** — it is a learning showcase and an unseen-course
  fallback, not a guaranteed fastest lap.
- The TAS path is deterministic and fast but only applies to a **known** course.

See `docs/TROUBLESHOOTING.md` for common fixes and `docs/DESIGN_NOTES.md` for the
reasoning behind these choices.
