# Event Handling

Events let you react to model conditions during simulation by executing actions and logging whenever the `cond` expression becomes true. They run either before the step (`phase = "pre"`), after (`phase = "post"`), or both, and you can attach logging to capture diagnostics for debugging or analysis. 

NOTE: Some fast-path runners for analysis prefer models without events.

## Basic Template

```toml
[events.reset_on_threshold]
phase = "post"
cond = "x > threshold"
action = "x = 0; spike_count = spike_count + 1"
log = ["t", "x", "spike_count"]
```

- `phase` controls when the condition is evaluated (default is `post`).
- `cond` must be a string returning a boolean. It is re-evaluated every timestep.
- `action` is a string of assignment statements; you can also scope assignments as `action.var = "expr"` for clarity.
- `log` is optional and lists variables whose values are recorded when the event fires.

## Condition Context

- **States/parameters**: Reference any declared state or parameter.
- **Aux variables**: Reuse derived expressions from `[aux]` to keep conditions readable.
- **Time (`t`)**: Always available for time-based triggers.
- **User-defined functions**: Call them just like in equations or aux definitions.
- **Built-in math & scalar macros**: `sin`, `cos`, `clip`, `approx`, etc.
- **Generator comprehensions**: Use `sum(...)` or `prod(...)` when you need reductions.
- **Event macros**: `cross_up`, `cross_down`, `changed`, `in_interval`, `enters_interval`, `leaves_interval`, `increasing`, `decreasing`, and `cross_either` automatically compare lagged state values so you do not need to write manual `lag_` expressions.
- **Lag notation**: You can call `lag_state(k)` within conditions, but only for real state variables, never aux.

### Event Macros Example

```toml
[events.detect_spike]
phase = "pre"
cond = "cross_up(v, 1.0)"
action = "spike_count += 1"
```

The macro handles the lagged access for you, so the condition fires the instant `v` crosses the threshold from below without extra bookkeeping.

## Action Details

- Actions can modify states, parameters (if allowed), aux, or tracker variables by assigning new expressions.
- Use semicolons to separate multiple statements or define individual assignments with `action.var = "expr"` syntax.
- Actions execute atomically after the condition is evaluated; side effects become part of the model state for the next timestep.
- Keep actions short; heavy computations belong in aux variables or helper functions.

## Logging

- `log` captures the listed expressions whenever the event is triggered.
- Logs can include states, aux, or computed expressions (`log = ["t", "energy", "debug_flag"]`).
- Use logging to inspect event timing, detect spurious triggers, or record counters for analysis.

## Event Lifecycle with Mods

Mods can manipulate events using the same verbs available elsewhere:

- `mod.remove.events` deletes existing events by name.
- `mod.replace.events.name` redefines the phase/cond/action/log for an event that already exists.
- `mod.add.events.new_name` inserts a new event (errors if the name already exists).
- `mod.set.events` is not supported; use `add` or `replace` instead.

Always remember the global verb order: remove → replace → add → set, so you can remove or replace an event before adding another with the same identifier.

## Best Practices

1. **Name events descriptively** (`events.detect_refractory_start`) so their intent is clear.
2. **Extract complex predicates into aux variables or functions** to keep conditions readable.
3. **Keep actions small and deterministic**, and prefer updating derived quantities via aux rather than inlined expressions.
4. **Use event macros when tracking crossings or changes** to avoid manual lag bookkeeping.
5. **Log intentionally**—too many log entries can degrade performance, so record only what you need for debugging or analysis contexts.
