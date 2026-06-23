# Seen-Course TAS Plan

The seen course is known in advance, so the highest-value, most reliable points
come from a **Tool-Assisted run (TAS)**: a fixed, optimized input sequence that is
replayed deterministically. The same file produces the same lap every time, which
is exactly what the scoring rewards.

## Points this targets

| Pts | Requirement | How the TAS delivers it |
|----|-------------|--------------------------|
| 20 | Complete seen course + publish time | Replay any finishing inputs file |
| 5  | Complete seen course **without human input** | A replay has no human input |
| 15 | Beat the **judges'** time (no human input) | Bruteforce drives the time down |
| 10→1 | Top-10 leaderboard (seen, fastest) | Fastest optimized run |

## Status

- **Baseline finishing run: DONE.** `tas/bytesandbolts.txt` is a valid TMInterface
  input script (`start-end press up/left/right`) that finishes at ~54.7s. Replaying
  this already banks the 20 + 5 point criteria.
- **Remaining: optimize it** with bruteforce to chase the beat-time and leaderboard
  points, then publish.

## What a TAS is (vs. the live agent)

`scripts/run_live_agent.py` decides inputs live from the screen, in real time. A TAS
is the opposite: an **offline-optimized, fixed input list** for one known track,
replayed exactly. The live agent is the *unseen-course* fallback; the TAS is the
*seen-course* primary.

## Prerequisites

- TMNF launched through **TMInterface 1.4.3**.
- The seen course loaded (`BytesAndBoltsCourse1.Challenge.Gbx`).
- The project venv active.

## Step 1 — Replay the baseline (locks in 20 + 5)

In the TMInterface console (open it in-game):

```
load tas/bytesandbolts.txt
```

Let it replay to the finish. Confirm it completes and a time appears, then publish
that time to the platform. This is your safety net — bank it early.

> Paths are relative to the TMInterface working directory. If `load` cannot find the
> file, use an absolute path to `tas/bytesandbolts.txt`.

## Step 2 — Bruteforce to go faster (chases 15 + 10)

1. Load the baseline and enable continuous search in the TMInterface console:

   ```
   load tas/bytesandbolts.txt
   set bf_search_forever true
   ```

   Optional tuning (see TMInterface docs / donadigo.com/tmtas):
   - `set bf_max_steer_diff <n>` — how far steering can be nudged per mutation.
   - `set bf_max_time_diff <n>` — how far input timings can shift.
   - The default target is **finish time**, which is what we want.

2. Start the optimizer:

   ```
   bruteforce
   ```

3. In a terminal, run the monitor so improved inputs are saved automatically:

   ```powershell
   python scripts/tas_bruteforce.py --output tas/course1_optimized.txt
   ```

   Each time the optimizer accepts a faster solution, the script writes the new
   inputs to `tas/course1_optimized.txt`. Leave it running (minutes → hours);
   longer search generally means a faster time.

4. Stop the search in-game when satisfied. `tas/course1_optimized.txt` holds the
   best inputs found.

## Step 3 — Hand-tune hard sections (optional)

Where bruteforce stalls, use TMInterface's TAS editor / save-states to manually
adjust the inputs around a tricky corner or jump, then resume bruteforce from there.

## Step 4 — Submit

1. Fresh-load the seen course.
2. `load tas/course1_optimized.txt` and let it replay to the finish.
3. Confirm the time is reproducible (run it twice — it should be identical).
4. Publish the time to the online platform.

## Verification checklist

- [ ] `python scripts/01_smoke_tminterface.py` connects (bridge works).
- [ ] Baseline `tas/bytesandbolts.txt` replays to a clean finish.
- [ ] `scripts/tas_bruteforce.py` connects and writes `tas/course1_optimized.txt`
      when the optimizer improves.
- [ ] Optimized inputs replay to a clean, reproducible finish.
- [ ] Time published to the platform.

## Notes / risks

- Bruteforce needs a *finishing* seed run; we have one (`tas/bytesandbolts.txt`).
- Confirm with organizers that a TMInterface-validated replay/time satisfies the
  "no human input" criterion before relying on it.
- Keep `run_live_agent.py` runnable — it is the unseen-course insurance and the ML
  talking point for judges.
