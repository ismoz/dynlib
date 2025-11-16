#!/usr/bin/env python3
"""
This script demonstrates the new path resolution and URI system,
showing various ways to load and run models.
"""
from dynlib import setup


def demo_inline_model():
    """Demo 1: Load a model from inline TOML."""
    print("=" * 60)
    print("Demo 1: Inline Model Definition")
    print("=" * 60)
    
    inline_model = """
    inline:
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
    
    uri = inline_model
    sim = setup(uri, stepper="euler", jit=False)
    
    print(f"Model kind: {sim.model.spec.kind}")
    print(f"States: {sim.model.spec.states}")
    print(f"Stepper: {sim.model.stepper_name}")
    
    # Run simulation
    sim.run(T=2.0)
    results = sim.results()
    
    print(f"Simulation ran {len(results)} steps")
    print(f"Initial x: {results['x'][0]:.6f}")
    print(f"Final x: {results['x'][-1]:.6f}")
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
    sim = setup(str(model_path), stepper="euler", jit=False)
    
    print(f"Loaded from: {model_path}")
    print(f"Model kind: {sim.model.spec.kind}")
    print(f"States: {sim.model.spec.states}")
    print(f"Default stepper: {sim.model.stepper_name}")
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
    print("║" + " " * 10 + "DYNLIB URI System Demo" + " " * 26 + "║")
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
