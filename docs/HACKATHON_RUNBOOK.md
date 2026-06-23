# Hackathon Runbook

## Before judging

1. Set TMNF to windowed `1280x960`.
2. Auto-hide the Windows taskbar.
3. Launch TMNF through **TMInterface 1.4.3**.
4. Use normal behind-car camera.
5. Open PowerShell in the project root:

```powershell
.venv\Scripts\activate
```

6. Optional but recommended:

```powershell
python scripts/02_check_capture.py --config config/default.yaml
```

Confirm that the road/open area is highlighted and pseudo-LIDAR rays are visible.

## During the 5-minute setup

1. Let judges load the unseen map.
2. Put car on the start line.
3. Make sure TMNF window is visible and not covered.
4. Run:

```powershell
python scripts/run_live_agent.py --config config/default.yaml
```

5. Move the debug window so it does not cover the game window.
6. Press **F8** when ready to hand control to the agent.

## During attempts

- Press **F8** to pause/resume.
- Press **Esc** for emergency stop.
- If the car is badly stuck, manually respawn yourself.
- If the agent is too aggressive, lower speeds in `config/default.yaml`.

## Emergency fallback

If the vision debug is clearly wrong, run slower:

```yaml
controller:
  max_speed_kmh: 70
  corner_speed_kmh: 45
  unsure_speed_kmh: 25
```

If the agent cannot find the game window:

```powershell
python scripts/list_windows.py
```

Then update:

```yaml
window:
  title_substring: "whatever title you see"
```
