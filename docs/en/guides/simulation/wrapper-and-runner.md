# Wrapper and Runner Interaction

The main simulation scheme is organized as follows: wrapper ⊃ runner ⊃ stepper. Runner and stepper form a jittable simulation kernel. This kernel is designed for numba compatibility, so it runs in a tight loop without complex python operations. `emit()` method of the stepper provides the stepper side calculator of this kernel. It returns a jittable function. At each simulation step runner calls this function. Stepper function has its own internal loop. It can retry until a successful next step value is obtained. At each step, runner checks the status of stepper result, buffers, recordings etc. If an unusual event occurs before the simulation finishes, it can pause the simulation and return all responsibility to the wrapper with a status code. Wrapper investigtates the status code and performs the necessary action that can't be performed with a jittable kernel (like buffer reallocation). After the necessary action wrapper restarts the runner (unless the status is terminal).

Below the details of this scheme are explained.

## Responsibilities in the stack
- `Sim._execute_run` prepares seeds, recording selections, stop-phase masks, and workspace hints before calling `run_with_wrapper`.  The wrapper therefore sees the same inputs that `Sim.run` would, along with the compiled callables and `[sim]` defaults it needs to drive execution (`src/dynlib/runtime/sim.py:2339`).
- `run_with_wrapper` then becomes the conductor: it reserves recording/event buffers, workspaces, observer hooks, and stepping parameters before every runner invocation (`src/dynlib/runtime/wrapper.py:34`).  This keeps the hot runner loop lean and focused on stepping, while the wrapper handles setup, growth, and post-processing.

## What the wrapper does
1. **Allocate and seed every workspace.** Runtime and stepper workspaces are created once and optionally seeded from `seed.workspace` to enable resume scenarios.
2. **Manage recording & event pools.** The wrapper slices recording arrays just for the selected state/aux indices, allocates event-log buffers with the requested `max_log_width`, and keeps cursor ints for runner results.
3. **Triage execution configuration.** It translates `Sim` knobs (dt, adaptive flags, discrete horizons, WRMS picks, stop phases, observer modules) into the inputs the runner needs, including the initial `dt_curr`, stop-phase mask, and which runner variant to load.
4. **Wire up analysis & observers.** If observers are registered, the wrapper allocates their workspaces, trace buffers, and variational hooks, then packages metadata so the rest of the runtime can e.g. add diagnostics after the runner returns.
5. **Drive the runner loop.** Inside `while True`, it repeatedly calls the compiled runner, passes the buffers it owns, and responds to the status codes the runner returns.

## Runner ABI contract
The runner is a JIT-friendly callable that knows nothing about recordings, observers, or buffer growth; it only obeys the frozen ABI defined in `runner_api.py`.  Each call receives: scalars like `(t0, t_end, dt_init, max_steps)`, model storage (`y_curr`, `params`), workspaces, recording buffers, event logs, analysis slots, and low-level cursors/caps.  It drives a single epoch of stepping and may only exit through the well-defined statuses.

Important statuses:
- `DONE` / `EARLY_EXIT` tell the wrapper the horizon was reached (or a stop condition triggered).  The wrapper then copies the final state and queues a `Results` object.
- `GROW_REC` / `GROW_EVT` tell the wrapper it needs to resize the recording or event buffers, re-enter the runner with updated caps, and keep the cursors returned so the runner can resume where it left off.
- `STEPFAIL`, `NAN_DETECTED`, `USER_BREAK`, `TRACE_OVERFLOW` bubble back as warnings; the wrapper still builds a `Results` snapshot so callers can inspect partial output.
- `OK` stays internal; the wrapper keeps re-invoking the runner until one of the exiting statuses is seen.

## Looping and re-entry
When the runner reports growth, the wrapper:
- enlarges the appropriate buffer while copying existing data, updates the cap,
- rewinds the start cursors (`i_start`, `step_start`), and
- uses the last committed time/dt so the runner resumes the next chunk without losing continuity.

If the runner finishes normally, the wrapper snapshots the workspaces, final state/dt, recorded arrays, event logs, and observer trace info before returning `Results` to `Sim`.

## Data flow summary
1. `Sim._execute_run` packages up the state seed and `[sim]` knobs and hands them to `run_with_wrapper`.
2. The wrapper prepares buffers, analysis metadata, and selective recording lists, then calls the compiled runner (which consumes `stepper`, `rhs`, event-callables, and the data buffers).
3. The runner returns a status; the wrapper interprets it, grows buffers if needed, and loops until the run completes or aborts.
4. Finally, the wrapper builds a `Results` instance with the recorded arrays, event logs, traces, and final state so that `Sim.results()` can return user-friendly views.

This separation keeps the handwritten wrapper responsible for Python-side bookkeeping while letting the compiled runner concentrate on the numerically intensive stepping loop.
