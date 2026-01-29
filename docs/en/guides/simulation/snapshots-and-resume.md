# Snapshots & Resume

`Sim` keeps a live `SessionState` that encapsulates time, states, parameters, the stepper workspace, and runtime metadata. Snapshots capture that state at a point in time so you can rewind, branch, or serialize a simulation, while `run(resume=True)` lets you grow recorded segments without rebuilding the model from scratch.

## Snapshot fundamentals

- **Initial snapshot**: The `"initial"` snapshot is created automatically before the first `run()` so you always have a known starting point to fall back to.
- **`create_snapshot(name, description="…")`**: Clones the current `SessionState`, records the current `time_shift`/`dt`, stamps the snapshot with `name`/`description`, and saves the full workspace + stepper config. Duplicate names raise an error, so pick descriptive identifiers.
- **`list_snapshots()`** returns `name`, simulation time `t`, step count, creation timestamp, and any description you provided.
- **`compat_check(snapshot)`** compares `SessionPins` (spec hash, stepper, workspace signature, dtype, dynlib version) to guarantee a snapshot comes from a compatible build. `reset()` uses the same guard and will fail fast when the model, stepper, or dtype has changed.

Snapshots are lightweight to create and cheap to keep around, so take one whenever you hit a milestone (e.g., after applying a stimulus, finishing a parameter sweep, etc.).

## Resetting and restarting

- **`reset(name="initial")`** rolls the session back to a named snapshot and wipes recorded history, segments, pending run tags, and resume state. It restores `_time_shift` and `_nominal_dt` from the snapshot so that future `run()` calls start from that exact moment.
- After `reset`, the recorder is cleared, which also resets the stored `record_vars` selection, so you can choose a different subset of variables before the next run.
- `session_state_summary()` reports `can_resume`/`reason`, letting you query whether `resume=True` is allowed without triggering the run logic.

## Running and resuming

`Sim.run(resume=True)` continues from the most recent `SessionState` instead of restarting integration from `t0`. Key behaviors:

1. **Session continuity**: The workspace, stepper configuration, and `time_shift` from the previous run are preserved, so the next segment feels like an uninterrupted extension in both deterministic and adaptive steppers.
2. **Recording constraints**: You cannot override `ic`, `params`, `t0`, or `dt`, so `resume` always starts where the session left off. Warm-up `transient > 0` is disallowed because resume segments must continue immediately. `record_vars` cannot be re-specified either; the first recorded run after a `reset` fixes the variable list and every subsequent `resume` run reuses that list automatically.
3. **Segment tracking**: Each recorded run appends a `Segment` entry describing `t_start`, `t_end`, `step_start`, `step_end`, and whether the chunk was produced via resume. Pass `tag="label"` to the `run()` call to assign a friendly name—`run#0`, `run#1`, etc. are generated otherwise. Rename with `name_segment()` or `name_last_segment()` when you need human-readable labels for `ResultsView`.
4. **Results stitching**: Resume reuses the same `_ResultAccumulator`, so `raw_results()`/`results()` see a seamless time-series spanning every segment. `run(resume=True)` throws if the requested horizon does not extend beyond the current time, avoiding overlaps.
5. **Compatibility guard**: Before resuming, `can_resume()` compares the current pins to those captured in the `SessionState`. If it returns `(False, reason)`, rewrite the simulation by calling `reset()` or rebuild with a compatible `FullModel`.
6. **No parameter overrides inside resume**: For a resumed segment you cannot pass new `ic`, `params`, `t0`, or `dt`. The run keeps the parameters, stepper workspace, and timing from the previous segment, so changing values requires a reset, snapshot, or a separate `run()` call that does not set `resume=True`.

Typical pattern:

```python
sim.run(T=2.0, record=True, tag="phase-1")
sim.create_snapshot("phase-1", "after the first stimulus")

# Continue without rebuilding; the second run is appended
sim.run(T=5.0, resume=True, tag="phase-2")

# Start a different branch by resetting to the saved snapshot
sim.reset("phase-1")
sim.run(T=3.0, record=True, tag="phase-1-replay")
```

If you need to change parameters between segments, do so before the resumed run: reset to an earlier snapshot, `assign()` the new parameter/state values (or import a snapshot that already encodes them), then run without `resume` or call `run(resume=True)` once the new values are in place. Resume never accepts `ic`, `params`, `t0`, or `dt` overrides, so any new configuration must be done via snapshots/assignments that happen before the resumed segment starts.

### Examples of changing parameters between segments

**Example 1: Branching with parameter changes (no resume)**

```python
# Run initial segment
sim.run(T=2.0, record=True, tag="baseline")

# Create snapshot at end of first segment
sim.create_snapshot("after-baseline", "End of baseline run")

# Reset to snapshot and change a parameter
sim.reset("after-baseline")
sim.assign(I=15.0)  # Change input current parameter

# Run new segment with modified parameter (starts fresh recording)
sim.run(T=3.0, record=True, tag="modified-current")
```

**Example 2: Continuing with parameter changes (using resume)**

```python
# Run first segment
sim.run(T=2.0, record=True, tag="phase-1")

# Create snapshot
sim.create_snapshot("phase-1-end", "End of phase 1")

# Reset and modify parameters
sim.reset("phase-1-end")
sim.assign(a=0.02, b=0.25)  # Change Izhikevich parameters

# Continue from the reset point with new parameters
sim.run(T=5.0, resume=True, tag="phase-2-modified")
```

**Example 3: Using `assign()` with `clear_history=True` to start fresh**

The `assign()` method has an optional `clear_history` parameter. When `clear_history=True`, it clears all previous results and segments while preserving the current session state (time, workspace, etc.) with the new assigned values. This effectively allows you to create a new segment without resetting to a snapshot:

```python
# Run initial segment
sim.run(T=2.0, record=True, tag="initial")

# Assign new parameter values and clear history to start fresh recording
sim.assign(I=20.0, clear_history=True)

# This run creates a new segment (since history was cleared)
sim.run(T=3.0, record=True, tag="new-segment")
```

Note that `clear_history=True` does not change the simulation time or workspace state—it only clears the recorded results, allowing the next `run()` to start a new segment from the current session state.

## Persistence and portability

- **`export_snapshot(path, source="current" | "snapshot", name=...)`** writes a `.npz` file containing:
  - `meta.json` (schema version, pins, names, time/step counters, `time_shift`, `nominal_dt`, stepper config names/values)
  - `y` and `params` vectors
  - Workspace buckets (`workspace/runtime/<name>`, `workspace/stepper/<name>`) and `stepper_config` if present
  - The write happens atomically via a temporary file so partial writes never corrupt existing snapshots.
- **`inspect_snapshot(path)`** reads `meta.json` without changing the session, letting you verify compatibility before importing.
- **`import_snapshot(path)`** replaces the current session state with the contents of a snapshot file, clears results/segments, resets `_time_shift`/`_nominal_dt`, and rejects files whose pins do not match the active model.

Use export/import when you need to checkpoint a long computation, share a resume point with a colleague, or persist workflow state between CI runs.

## Segment naming and resume metadata

- Each `Segment` carries a `cfg_hash` (stepper configuration digest) plus `resume` flag so downstream tools can tell whether a chunk was created during a fresh run or via `resume`.
- Use `tag` on `run()` or `name_segment()` afterward to keep your `segments` listing readable; `ResultsView` exposes these names so you can quickly extract the portion you care about.
- When you reset, the segment list empties, but snapshots remain. Continue appending segments only with `resume=True`—otherwise `run()` clears the accumulator and starts a new recording pass.

Snapshot and resume controls keep your experiments reproducible: take snapshots at branch points, reset to them when you need to test variations, and resume to build long trajectories without losing history.
