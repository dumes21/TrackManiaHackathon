# TMNF Live Unseen Agent v1

A practical, hackathon-focused autonomous driver for **TrackMania Nations Forever**.

This version is built for the unseen-course challenge where the map is loaded live during testing. It does **not** use SAC/RL training, GBX parsing, pretrained CNNs, or behaviour cloning. It uses:

- **TMInterface 1.4.3** for clean input injection and telemetry
- **screen capture** of the TMNF window
- **OpenCV road/open-path detection**
- **pseudo-LIDAR rays** from the screen image
- a conservative **rule-based controller**
- **F8** start/pause and **Esc** emergency stop
- a debug window showing what the agent sees

The goal is not to set a world-record lap. The goal is to get a solid autonomous system running quickly and give it the best chance of completing an unseen track safely.

---

## 1. Required setup

Use this exact setup for consistency:

```text
Windows
TrackMania Nations Forever
TMInterface 1.4.3
TMNF windowed at 1280x960
Windows taskbar auto-hidden
Normal behind-car camera
```

Do **not** launch TMNF normally. Launch it through **TMInterface 1.4.3**.

---

## 2. Install Python dependencies

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Run order

### Step 1 — Smoke test TMInterface

Open TMNF through TMInterface 1.4.3, load any map, sit on the start line, then run:

```powershell
python scripts/01_smoke_tminterface.py
```

Expected result:

```text
[tm] Registered to server: TMInterface0
[smoke] Connected. Holding accelerate...
speed=...
pos=...
yaw=...
```

The car should accelerate straight. It may crash; that is fine. This only proves the bridge works.

### Step 2 — Check screen capture and vision

Keep TMNF open and visible. Run:

```powershell
python scripts/02_check_capture.py --config config/default.yaml
```

You should see a debug view with:

- green-ish detected road/open-path mask
- pseudo-LIDAR rays
- a magenta target direction
- a cyan seed box near the lower centre

Press `q` or `Esc` in the debug window to quit.

### Step 3 — Run the live agent

Load the map, sit on the start line, then run:

```powershell
python scripts/run_live_agent.py --config config/default.yaml
```

Controls:

```text
F8  = start / pause autonomous control
Esc = emergency stop and release inputs
q   = quit if debug window is focused
```

Important: keep the debug window **away from the TMNF window**. If the debug window covers the game, screen capture may capture the debug window instead of the road.

---

## 4. How the agent drives

Every control tick, the agent does this:

```text
capture TMNF window
    ↓
crop road-ahead region
    ↓
build road/open-path mask
    ↓
cast pseudo-LIDAR rays into the mask
    ↓
combine road-centre error + best ray direction
    ↓
read speed from TMInterface
    ↓
choose steer / accelerate / brake
    ↓
send controls through TMInterface
```

The agent is intentionally conservative:

- it slows down when the road is unclear
- it slows down on sharp steering
- it brakes/reverses if stuck
- it does **not** auto-respawn

You manually respawn if needed.

---

## 5. Tuning knobs

Most useful settings are in `config/default.yaml`.

For safer driving, lower:

```yaml
controller:
  max_speed_kmh: 85
  corner_speed_kmh: 50
```

For more aggressive driving, raise:

```yaml
controller:
  max_speed_kmh: 125
  corner_speed_kmh: 80
```

If the agent steers too weakly:

```yaml
controller:
  steer_kp: 1.35
```

If it wiggles too much:

```yaml
controller:
  steer_kd: 0.15
```

If it keeps thinking the road is blocked:

```yaml
controller:
  low_front_clearance: 0.25
  medium_front_clearance: 0.40
```

---

## 6. Files

```text
config/default.yaml                 Main settings
scripts/01_smoke_tminterface.py     Proves Python ↔ TMInterface works
scripts/02_check_capture.py         Shows live capture + vision debug
scripts/run_live_agent.py           Main autonomous agent
scripts/list_windows.py             Lists visible windows if capture cannot find TMNF
tm_live_agent/tminterface_bridge.py TMInterface telemetry/input bridge
tm_live_agent/window_capture.py     TMNF window capture via mss + pywin32
tm_live_agent/vision.py             Road mask + pseudo-LIDAR
tm_live_agent/controller.py         Speed-aware rule controller
tm_live_agent/debug_view.py         Debug overlay
tm_live_agent/hotkeys.py            F8 / Esc handling
docs/HACKATHON_RUNBOOK.md           5-minute judging checklist
docs/TROUBLESHOOTING.md             Common fixes
```

---

## 7. Known limitations

This is a quick-solid v1, not a magic Trackmania solver.

It may struggle with:

- loops
- wallrides
- jumps where the road disappears
- weird lighting/road surfaces
- off-road sections
- tracks where the visible road does not indicate the correct route

But it is intentionally simple, fast, debuggable, and independent of the unseen `.gbx` file.
