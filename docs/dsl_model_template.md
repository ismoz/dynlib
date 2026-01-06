# DSL Model File Template

This is a quick reference template for creating DSL model files in TOML format. 
It lists all available tables and their keys.

## Required Tables

### [model]
- `type` (required): "ode" | "map"
- `label` (optional): string
- `dtype` (optional): data type, default "float64"

### [states]
- `state_name = initial_value` (order defines the authoritative state vector order)
- For value expressions like 8/3 use quotes: "8/3".

## Optional Tables

### [constants]
- `constant_name = value` (scalars, numeric expressions allowed, can reference prior constants)
- Constants cannot be assigned and they are read-only literals.

### [params]
- `param_name = value` (scalars or arrays, cast to model dtype)

### Equations (choose one form or mix)
#### [equations.rhs] (per-state form)
- `state_name = "expression"`

#### [equations] (block form)
- `expr = """dx = expression \n dy = expression"""`

### [equations.jacobian] (optional dense Jacobian)
- `expr = [[ "...", "...", ... ], [...], ...]` (n × n matrix of expressions)
- State vector order is the [states] declaration order (after mods). For `state_names = (s0, s1, ...)`, `expr[i][j]` is ∂f_state_names[i]/∂state_names[j]. Reordering [states] is a semantic change and changes how matrix literals are interpreted.

### [aux]
- `aux_name = "expression"`

### [functions.function_name]
- `args = ["arg1", "arg2", ...]`
- `expr = "expression"`

### [events.event_name]
- `phase` (optional): "pre" | "post" | "both" (default "post")
- `cond = "expression"`
- `action = "expression"` or `action.state_name = "expression"`
- `tags` (optional): ["tag1", "tag2", ...]
- `log` (optional): ["var1", "var2", ...]

### [sim]
- `t0 = value`
- `t_end = value`
- `dt = value`
- `stepper = "euler" | "rk4" | ...`
- `record = true/false`
- `stepper_config = value (stepper-specific config values)`

### [meta]
- `title = "string"`
