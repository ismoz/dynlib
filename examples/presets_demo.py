#!/usr/bin/env python3
"""
Demo of the presets feature in dynlib v2.

Shows:
- Inline presets defined in the model DSL
- Listing and applying presets
- Loading presets from external files
- Saving presets to files
- Round-trip preservation
"""

import tempfile
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.compiler.build import build
from dynlib.runtime.sim import Sim


# Model with inline presets
MODEL_TOML = """
[model]
type = "ode"
label = "Izhikevich Neuron (Simple)"

[states]
v = -65.0
u = -13.0

[params]
a = 0.02
b = 0.2
c = -65.0
d = 8.0
I = 10.0

[equations.rhs]
v = "0.04*v*v + 5*v + 140 - u + I"
u = "a*(b*v - u)"

[events.spike]
phase = "post"
cond = "v >= 30"
log = ["t"]

[events.spike.action]
v = "c"
u = "u + d"

[sim]
t0 = 0.0
t_end = 200.0
dt = 0.25
stepper = "euler"
record = true

# Inline presets for different neuron behaviors
[presets.regular_spiking.params]
a = 0.02
b = 0.2
c = -65.0
d = 8.0
I = 10.0

[presets.fast_spiking.params]
a = 0.1
b = 0.2
c = -65.0
d = 2.0
I = 10.0

[presets.bursting.params]
a = 0.02
b = 0.2
c = -50.0
d = 2.0
I = 15.0

[presets.bursting.states]
v = -70.0
u = -14.0
"""


def main():
    # Load model
    print("=== Loading model ===")
    doc = tomllib.loads(MODEL_TOML)
    spec = build_spec(parse_model_v2(doc))
    model = build(spec, stepper=spec.sim.stepper, jit=True)
    sim = Sim(model)
    
    # List inline presets
    print("\n=== Inline presets ===")
    presets = sim.list_presets()
    print(f"Available presets: {presets}")
    
    # Apply a preset and run
    print("\n=== Applying 'regular_spiking' preset ===")
    sim.apply_preset("regular_spiking")
    sim.run(t_end=100.0)
    results = sim.results()
    event_counts = results.event.summary()
    n_spikes = event_counts.get('spike', 0)
    print(f"Regular spiking: {n_spikes} spikes in 100ms")
    
    # Apply different preset
    print("\n=== Applying 'fast_spiking' preset ===")
    sim.reset()  # Reset to initial state
    sim.apply_preset("fast_spiking")
    sim.run(t_end=100.0)
    results = sim.results()
    event_counts = results.event.summary()
    n_spikes = event_counts.get('spike', 0)
    print(f"Fast spiking: {n_spikes} spikes in 100ms")
    
    # Save presets to file
    with tempfile.TemporaryDirectory() as tmpdir:
        preset_file = Path(tmpdir) / "neuron_presets.toml"
        
        print(f"\n=== Saving presets to {preset_file.name} ===")
        sim.reset()
        sim.apply_preset("regular_spiking")
        sim.save_preset("regular_spiking", preset_file)
        
        sim.apply_preset("fast_spiking")
        sim.save_preset("fast_spiking", preset_file)
        
        # Check file contents
        with open(preset_file, "rb") as f:
            doc = tomllib.load(f)
        print(f"File contains: {list(doc['presets'].keys())}")
        
        # Create a new sim and load from file
        print("\n=== Loading presets from file ===")
        sim2 = Sim(model)
        count = sim2.load_preset("*", preset_file, on_conflict="replace")
        print(f"Loaded {count} presets")
        print(f"Available: {sim2.list_presets()}")
        
        # Verify they work
        sim2.apply_preset("regular_spiking")
        sim2.run(t_end=100.0)
        results2 = sim2.results()
        event_counts2 = results2.event.summary()
        n_spikes2 = event_counts2.get('spike', 0)
        print(f"Loaded preset produces {n_spikes2} spikes (should match original)")
    
    # Demonstrate glob patterns
    print("\n=== Glob pattern matching ===")
    print(f"All presets: {sim.list_presets('*')}")
    print(f"Presets starting with 'fast': {sim.list_presets('fast_*')}")
    print(f"Presets ending with 'spiking': {sim.list_presets('*_spiking')}")
    
    print("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
