# Troubleshooting

## `Could not connect to TMInterface`

Check:

- TMNF is launched through **TMInterface 1.4.3**.
- The game is already running before the Python script starts.
- You are using server name `TMInterface0` unless you launched another server index.

Run:

```powershell
python scripts/01_smoke_tminterface.py
```

first. Do not debug the full agent until the smoke test works.

## `Could not find a visible window containing 'TmForever'`

Run:

```powershell
python scripts/list_windows.py
```

Find the actual TMNF window title and set it in `config/default.yaml`:

```yaml
window:
  title_substring: "TmForever"
```

## Debug window shows wrong image

The debug window may be covering the TMNF window. Move the debug window away from the game.

Screen capture grabs visible screen pixels. If another window covers TMNF, the capture can be corrupted.

## Road mask is poor

Try first:

- use the normal behind-car camera
- keep the same lighting/view
- make sure the car is on the road when starting

Then tune `config/default.yaml`:

```yaml
vision:
  adaptive_sat_tol: 70
  adaptive_val_tol: 90
  generic_max_saturation: 120
```

If it includes too much wall/sky, lower the tolerances:

```yaml
vision:
  adaptive_sat_tol: 40
  adaptive_val_tol: 55
  generic_max_saturation: 75
```

## Agent steers the wrong way

Try lowering ray influence and raising centre influence:

```yaml
controller:
  ray_bias: 0.35
  center_bias: 0.65
```

Or if it needs to choose open space more aggressively:

```yaml
controller:
  ray_bias: 0.70
  center_bias: 0.30
```

## Agent is too fast

Lower:

```yaml
controller:
  max_speed_kmh: 80
  corner_speed_kmh: 50
  unsure_speed_kmh: 25
```

## Agent wiggles

Lower derivative and steering gain:

```yaml
controller:
  steer_kp: 0.95
  steer_kd: 0.10
```

## Agent is too timid

Raise speeds slightly:

```yaml
controller:
  max_speed_kmh: 120
  corner_speed_kmh: 75
```

Do this only after it can survive at lower speed.
