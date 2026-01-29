# Mods: Model Modifications in Dynlib

Mods (modifications) in Dynlib allow you to dynamically alter model specifications without changing the original model files. This is useful for:

- Creating model variants (e.g., different parameter sets, added events)
- A/B testing different model configurations
- Applying patches or fixes to existing models
- Building complex models from simpler base models

## Overview

Mods are defined using TOML table syntax. A basic mod looks like:

```toml
[mod]
name = "my_modification"
group = "optional_group"
exclusive = false

[mod.remove.events]
names = ["event_to_remove"]

[mod.add.events.new_event]
phase = "post"
cond = "x > threshold"
action = "x = 0"

[mod.set.params]
alpha = 0.5
beta = 2.0
```

## Verb Operations

Mods support four main operations (verbs) that are applied in this order: **remove → replace → add → set**.

### 1. Remove

Removes existing components from the model. Only works on components that already exist.

**Supported targets:** `events`, `params`, `aux`, `functions`

```toml
[mod]
name = "cleanup"

# Remove specific events
[mod.remove.events]
names = ["debug_event", "temporary_trigger"]

# Remove parameters
[mod.remove.params]
names = ["unused_param", "legacy_constant"]

# Remove auxiliary variables
[mod.remove.aux]
names = ["temp_var", "debug_output"]

# Remove functions
[mod.remove.functions]
names = ["helper_func", "unused_util"]
```

**Note:** Attempting to remove `states` or other unsupported targets will raise an error.

### 2. Replace

Replaces existing components with new definitions. The component must already exist.

**Supported targets:** `events`, `aux`, `functions`

**Note:** Use `set.params` to update parameter values. Use `remove` + `add` to replace parameters entirely.

```toml
[mod]
name = "update_logic"

# Replace an event
[mod.replace.events.existing_event]
phase = "post"
cond = "x > new_threshold"
action = "x = 0; counter = counter + 1"

# Replace auxiliary variables
[mod.replace.aux]
energy = "0.5 * m * v^2"  # New expression
power = "force * velocity"  # New expression

# Replace functions
[mod.replace.functions.activation]
args = ["x", "gain", "offset"]
expr = "gain * tanh(x) + offset"
```

### 3. Add

Adds new components to the model. The component must not already exist.

**Supported targets:** `events`, `params`, `aux`, `functions`

```toml
[mod]
name = "add_features"

# Add new events
[mod.add.events.reset_mechanism]
phase = "post"
cond = "x > 10"
action = "x = 0"

[mod.add.events.spike_detector]
phase = "pre"
cond = "v > threshold"
action = "spike_count = spike_count + 1"
log = ["t"]  # Log spike times

# Add new parameters
[mod.add.params]
gain = 2.5
offset = 0.1

# Add auxiliary variables
[mod.add.aux]
total_energy = "kinetic + potential"
efficiency = "output_power / input_power"

# Add functions
[mod.add.functions.sigmoid]
args = ["x"]
expr = "1 / (1 + exp(-x))"

[mod.add.functions.relu]
args = ["x"]
expr = "max(0, x)"
```

**Note:** Attempting to add `states` or other unsupported targets will raise an error.

### 4. Set

Sets or updates component values. This is an "upsert" operation - it can create new components or update existing ones.

**Supported targets:** `states`, `params`, `aux`, `functions`

```toml
[mod]
name = "configure"

# Set state initial values
[mod.set.states]
x = 5.0
y = -2.5

# Set parameter values (must already exist)
[mod.set.params]
alpha = 0.1
beta = 2.5

# Set auxiliary variables (upsert - create or update)
[mod.set.aux]
debug = "t"  # Create new
energy = "0.5 * k * x^2"  # Update existing

# Set functions (upsert - create or update)
[mod.set.functions.activation]
args = ["x"]
expr = "tanh(x)"  # Update existing

[mod.set.functions.new_func]
args = ["a", "b"]
expr = "a + b"  # Create new
```

**Note:** For `states` and `params`, `set` only updates existing values and will raise an error if the component doesn't exist. Use `add` to create new parameters.

## Event Definition Format

Events in mods use the same TOML table format as model definitions:

```toml
[mod.add.events.event_name]
phase = "pre" | "post"        # When to check condition
cond = "expression"           # Condition to trigger
action = "code"               # Action to perform (string)
log = ["var1", "var2"]        # Variables to log when triggered (optional)

# Alternative: keyed action assignments
[mod.add.events.event_name]
phase = "post"
cond = "x > 5"
action.dx = 1.0
action.dy = -0.5
log = ["t"]
```

## Function Definition Format

Functions are defined with args and expr:

```toml
[mod.add.functions.function_name]
args = ["arg1", "arg2", "arg3"]  # Array of argument names
expr = "expression"              # Function body expression
```

## Group and Exclusivity

Mods can be grouped to prevent conflicting modifications:

```toml
# Exclusive mods in the same group
[mods.fast]
name = "fast"
group = "speed"
exclusive = true

[mods.fast.set.params]
dt = 0.01

[mods.slow]
name = "slow"
group = "speed"
exclusive = true

[mods.slow.set.params]
dt = 0.1

# Only one mod from the "speed" group can be active at a time
```

## Using Mods

### Loading Mods from Files

Mods are typically stored in TOML files and loaded via URI:

```python
from dynlib import build

# Load model with mods
model = build("model.toml", mods=["mods.toml#mod=variant1"])
```

#### Single Mod File

```toml
[mod]
name = "parameter_tune"
group = "tuning"

[mod.set.params]
alpha = 0.5
beta = 2.0

[mod.add.events.monitor]
phase = "post"
cond = "t % 1.0 == 0"
action = ""
log = ["x", "y"]
```

#### Multiple Mods in One File

```toml
[mods.variant1]
name = "variant1"

[mods.variant1.set.params]
gain = 1.0

[mods.variant2]
name = "variant2"

[mods.variant2.set.params]
gain = 2.0

[mods.variant2.add.events.noise]
phase = "pre"
cond = "true"
action = "x = x + 0.1 * randn()"
```

#### URI Patterns for Mods

- `"mods.toml"` - Load single mod from file
- `"mods.toml#mod=variant1"` - Load specific mod from collection
- `"inline: [mod]\nname='patch'\n..."` - Inline mod definition

### Programmatic Usage

For advanced use cases, you can create mods programmatically:

```python
from dynlib.compiler.mods import ModSpec, apply_mods_v2
from dynlib.dsl.parser import parse_model_v2

# Define mod as Python dict (equivalent to TOML above)
mod = ModSpec(
    name="programmatic_mod",
    set={
        "params": {"alpha": 0.5},
        "aux": {"debug": "t"}
    }
)

# Apply to parsed model
normal = parse_model_v2(model_toml_string)
modified = apply_mods_v2(normal, [mod])
```

## Error Handling

Mods validate operations and raise `ModelLoadError` for:

- **Unsupported targets**: Attempting to use operations on unsupported targets (e.g., `add.states`, `remove.states`, `replace.params`)
- **Non-existent components**: Attempting to remove or replace components that don't exist
- **Duplicate components**: Attempting to add components that already exist
- **Unknown components**: Attempting to set values for non-existent states or params
- **Invalid data types**: Using non-string values for aux variables
- **Malformed definitions**: Invalid function definitions (missing args, expr, etc.)
- **Group exclusivity violations**: Activating multiple exclusive mods from the same group

### Supported Targets by Verb

| Verb      | Supported Targets                    | Notes                                    |
|-----------|--------------------------------------|------------------------------------------|
| `remove`  | `events`, `params`, `aux`, `functions` | Component must exist                    |
| `replace` | `events`, `aux`, `functions`         | Component must exist                     |
| `add`     | `events`, `params`, `aux`, `functions` | Component must not exist                |
| `set`     | `states`, `params`, `aux`, `functions` | States/params must exist; aux/functions are upsert |

### Common Errors and Solutions

**Error: `add.states: unsupported target`**
- **Cause**: Trying to add new state variables via mods
- **Solution**: States cannot be added dynamically. Define them in the base model.

**Error: `remove.states: unsupported target`**
- **Cause**: Trying to remove state variables via mods
- **Solution**: States cannot be removed. They are fundamental to the model structure.

**Error: `replace.params: unsupported target`**
- **Cause**: Trying to replace parameters using `replace` verb
- **Solution**: Use `set.params` to update values, or use `remove.params` + `add.params` sequence.

**Error: `add.params.x: param already exists`**
- **Cause**: Trying to add a parameter that already exists
- **Solution**: Use `set.params` to update the value, or `remove.params` first if you need to replace it.

### Validation Prevents Silent Failures

Prior to validation improvements, unsupported operations would silently fail, leaving users confused about why their mods weren't working. Now, any attempt to use unsupported targets will immediately raise a clear error with a list of supported targets.

## Best Practices

1. **Use descriptive names**: Give mods clear, descriptive names that indicate their purpose.

2. **Group related mods**: Use groups for mutually exclusive options (e.g., different parameter sets).

3. **Test thoroughly**: Mods can significantly change model behavior - validate results carefully.

4. **Document mods**: Include comments explaining what each mod does and why.

5. **Version control**: Keep mod files under version control alongside your models.

6. **Start simple**: Begin with basic set operations, then progress to more complex add/replace/remove combinations.

## Examples

### Parameter Study Mods

```toml
[mods.low_gain]
name = "low_gain"
group = "gain_study"
exclusive = true

[mods.low_gain.set.params]
k = 0.1

[mods.high_gain]
name = "high_gain"
group = "gain_study"
exclusive = true

[mods.high_gain.set.params]
k = 10.0
```

### Replacing Parameters with Time-Varying Expressions

This example shows how to convert a constant parameter into a time-dependent auxiliary variable:

```toml
[mods.sine_drive]
name = "sine_drive"

# Note: Cannot add states via mods (would raise error)
# Instead, add auxiliary noise variables

[mods.sine_drive.add.params]
freq = 1000.0
Vmax = 4.0

# Add V as a time-varying auxiliary variable
[mods.sine_drive.add.aux]
V = "Vmax*sin(2*pi*freq*t)"
```

### Model Variants

```toml
[mod.stochastic]
name = "stochastic"

# Note: Cannot add states via mods (would raise error)
# Instead, add auxiliary noise variables

[mod.stochastic.add.params]
sigma = 0.1

[mod.stochastic.add.aux]
noise = "sigma * randn()"
noisy_x = "x + noise"
```

### Debugging Aids

```toml
[mod.debug]
name = "debug"

[mod.debug.add.aux]
debug_t = "t"
debug_x = "x"
debug_dx = "dx_dt"

[mod.debug.add.events.log_state]
phase = "post"
cond = "t % 1.0 == 0"  # Log every second
action = ""  # No action, just logging
log = ["debug_t", "debug_x", "debug_dx"]
```

### Function Variants

In the below example h(phi,N) function has two variants. One can be chosen as follows:

```python
sim = setup("memristive_chua#mod=odd")
```

```toml
[model]
type="ode"
name="Flux Controlled Memristor"

[states]
phi=0.1

[params]
a=0.08
b=1
c=0.83
d=1.8
N=0
freq=1.0
Vmax=4.0

[functions.W]
args=["phi"]
expr="a+b*tanh(phi)**2"

[aux]
V = "Vmax*sin(2*pi*freq*t)"
I = "W(phi)*V"

[equations.rhs]
phi="c*V-d*h(phi,N)"

### MODS:

[mods.odd.add.functions]
h = {args = ["phi","N"], expr="""
phi if N==0 else phi-sum(sign(phi+(2*j-1))+sign(phi-(2*j-1)) for j in range(1,N+1))
"""}

[mods.even.add.functions]
h = {args = ["phi","N"], expr = """
phi-sign(phi) if N==0 else phi-sign(phi)-sum(sign(phi+2*j)+sign(phi-2*j) for j in range(1,N+1))
"""}
```

## Verb Order Matters

Remember that verbs are applied in this fixed order:

1. **Remove** - Remove components first
2. **Replace** - Replace existing components  
3. **Add** - Add new components
4. **Set** - Set/update values

This allows complex transformations like: remove old component → add new one with different name, or replace component → then modify its parameters.
