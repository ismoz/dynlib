# examples/export_sources_demo.py
"""
Demonstration of compiled model source code export functionality.

This example shows how to:
1. Build a compiled model using setup()
2. Export the generated Python source code for inspection
3. Verify the compilation is correct
"""

from pathlib import Path
import tempfile
from dynlib import setup

def main():
    # Use an existing test model
    model_path = Path(__file__).parent.parent / "tests" / "data" / "models" / "decay.toml"
    
    # Setup simulation with JIT compilation
    print(f"Building model from: {model_path.name}")
    sim = setup(str(model_path), stepper="euler", jit=True, disk_cache=False)
    
    # Check if source code is available
    print(f"\nSource code availability:")
    print(f"  RHS source: {'✓' if sim.model.rhs_source else '✗'}")
    print(f"  Events pre source: {'✓' if sim.model.events_pre_source else '✗'}")
    print(f"  Events post source: {'✓' if sim.model.events_post_source else '✗'}")
    print(f"  Stepper source: {'✓' if sim.model.stepper_source else '✗'}")
    
    # Export sources to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir) / "compiled_model"
        print(f"\nExporting sources to: {export_dir}")
        
        exported_files = sim.model.export_sources(export_dir)
        
        print(f"\nExported files:")
        for component, filepath in exported_files.items():
            size = filepath.stat().st_size if filepath.exists() else 0
            print(f"  {component}: {filepath.name} ({size} bytes)")
        
        # Display the RHS source code
        if "rhs" in exported_files:
            print(f"\n{'='*60}")
            print("RHS Function Source Code:")
            print('='*60)
            rhs_content = exported_files["rhs"].read_text()
            print(rhs_content)
            print('='*60)
        
        # Display the stepper source code if available
        if "stepper" in exported_files:
            print(f"\n{'='*60}")
            print("Stepper Function Source Code:")
            print('='*60)
            stepper_content = exported_files["stepper"].read_text()
            print(stepper_content)
            print('='*60)
    
    print("\n✓ Export demo completed successfully!")

if __name__ == "__main__":
    main()
