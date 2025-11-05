#!/usr/bin/env python3
"""
Demo of Slice 6: URI-based Model Loading

This script demonstrates the new path resolution and URI system,
showing various ways to load and run models.
"""
from dynlib.compiler.build import build
from dynlib.runtime.sim import Sim
from dynlib.runtime.model import Model


def demo_inline_model():
    """Demo 1: Load a model from inline TOML."""
    print("=" * 60)
    print("Demo 1: Inline Model Definition")
    print("=" * 60)
    
    inline_model = """
[model]
type = "ode"

[states]
x = 1.0

[params]
a = 1.0

[equations.rhs]
x = "-a * x"

[sim]
t0 = 0.0
t_end = 2.0
dt = 0.1
stepper = "euler"
"""
    
    uri = f"inline: {inline_model}"
    full_model = build(uri, jit=False)
    
    print(f"Model kind: {full_model.spec.kind}")
    print(f"States: {full_model.spec.states}")
    print(f"Stepper: {full_model.stepper_name}")
    
    # Run simulation
    model = Model(
        spec=full_model.spec,
        stepper_name=full_model.stepper_name,
        struct=full_model.struct,
        rhs=full_model.rhs,
        events_pre=full_model.events_pre,
        events_post=full_model.events_post,
        stepper=full_model.stepper,
        runner=full_model.runner,
        spec_hash=full_model.spec_hash,
        model_dtype=full_model.model_dtype,
    )
    
    sim = Sim(model)
    results = sim.run()
    
    print(f"Simulation ran {results.n} steps")
    print(f"Initial x: {results.Y[0, 0]:.6f}")
    print(f"Final x: {results.Y[0, results.n-1]:.6f}")
    print()


def demo_file_loading():
    """Demo 2: Load a model from existing test file."""
    print("=" * 60)
    print("Demo 2: File-based Model Loading")
    print("=" * 60)
    
    from pathlib import Path
    data_dir = Path(__file__).parent.parent / "tests" / "data" / "models"
    model_path = data_dir / "decay.toml"
    
    if not model_path.exists():
        print(f"Test data not found: {model_path}")
        return
    
    # Load using absolute path
    full_model = build(str(model_path), jit=False)
    
    print(f"Loaded from: {model_path}")
    print(f"Model kind: {full_model.spec.kind}")
    print(f"States: {full_model.spec.states}")
    print(f"Default stepper: {full_model.stepper_name}")
    print()


def demo_uri_schemes():
    """Demo 3: Show different URI schemes."""
    print("=" * 60)
    print("Demo 3: URI Scheme Examples")
    print("=" * 60)
    
    examples = [
        ("inline: [model]\\ntype='ode'", "Inline TOML definition"),
        ("/abs/path/model.toml", "Absolute file path"),
        ("relative/model.toml", "Relative path from cwd"),
        ("model", "Extensionless (tries model.toml)"),
        ("proj://model.toml", "TAG resolution from config"),
        ("model.toml#mod=fast", "Fragment selector for mods"),
        ("TAG://path/to/model.toml#mod=variant", "Combined TAG + fragment"),
    ]
    
    print("Supported URI schemes:")
    for uri, description in examples:
        print(f"  {uri:40s} - {description}")
    print()
    
    print("Configuration:")
    print("  Config file: ~/.config/dynlib/config.toml (Linux)")
    print("  Or: ~/Library/Application Support/dynlib/config.toml (macOS)")
    print("  Or: %APPDATA%\\dynlib\\config.toml (Windows)")
    print()
    print("  Environment overrides:")
    print("    DYNLIB_CONFIG=/custom/config.toml")
    print("    DYN_MODEL_PATH=proj=/extra/path,/another")
    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "DYNLIB Slice 6 URI System Demo" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    demo_inline_model()
    demo_file_loading()
    demo_uri_schemes()
    
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
