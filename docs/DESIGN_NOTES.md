# Design Notes

## Why no SAC/RL in v1?

A pure RL agent needs a stable environment, reward shaping, many attempts, and careful tuning. For this hackathon, the unseen track is loaded live with limited setup and attempts, so the v1 agent is rule-based and reactive.

## Why no GBX parsing?

The plan assumes the unseen course is loaded inside the game and the team may not receive a `.gbx` file in a way that can be processed. Therefore, v1 uses only live screen input plus live TMInterface telemetry.

## Why pseudo-LIDAR?

True game-engine LIDAR would require map geometry access or a custom plugin/bridge. Pseudo-LIDAR is simpler: the agent detects likely drivable/open pixels in the screen image and casts rays through that mask.

This gives approximate signals:

- centre path clear or blocked
- left side more open
- right side more open
- road confidence low/high

## Why TMInterface input injection?

It is cleaner than keyboard automation. The bridge applies the latest command on every TMInterface physics callback using `set_input_state`, while the main loop updates commands at about 20 Hz.
