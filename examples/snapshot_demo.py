# examples/snapshot_demo.py
"""
Demonstration of snapshot export/import functionality in dynlib.

This example shows how to:
1. Export current simulation state to disk
2. Export named snapshots to disk  
3. Import snapshots from disk
4. Inspect snapshot metadata
"""

from pathlib import Path
import tempfile
import tomllib
from dynlib.dsl.parser import parse_model_v2
from dynlib.dsl.spec import build_spec
from dynlib.compiler.build import build
from dynlib.runtime.sim import Sim
from dynlib.runtime.model import Model


def create_model():
    """Create a simple exponential decay model for demonstration."""
    model_toml = """
[model]
type = "ode"
label = "Demo Exponential Decay"
dtype = "float64"

[states]
x = 1.0

[params]
decay_rate = 0.1

[equations]
"dx" = "-decay_rate * x"

[sim]
stepper = "euler"
dt = 0.01
t0 = 0.0
t_end = 10.0
record = true
"""
    
    data = tomllib.loads(model_toml)
    spec = build_spec(parse_model_v2(data))
    full_model = build(spec, stepper=spec.sim.stepper, jit=True)
    return Model(
        spec=full_model.spec,
        stepper_name=full_model.stepper_name,
        struct=full_model.struct,
        rhs=full_model.rhs,
        events_pre=full_model.events_pre,
        events_post=full_model.events_post,
        stepper=full_model.stepper,
        runner=full_model.runner,
        spec_hash=full_model.spec_hash,
        dtype=full_model.dtype,
    )


def main():
    print("=== Dynlib Snapshot Export/Import Demo ===\n")
    
    # Create simulation
    model = create_model()
    sim = Sim(model)
    print("✓ Created simulation with exponential decay model")
    
    # Run to some intermediate state
    sim.run(t_end=2.0)
    print(f"✓ Ran simulation to t=2.0, current state: x={sim._session_state.y_curr[0]:.4f}")
    
    # Create an in-memory snapshot
    sim.create_snapshot("checkpoint_1", "After running to t=2.0")
    print("✓ Created in-memory snapshot 'checkpoint_1'")
    
    # Continue running
    sim.run(t_end=5.0, resume=True)
    print(f"✓ Continued to t=5.0, current state: x={sim._session_state.y_curr[0]:.4f}")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Export current state
        current_snap_path = tmp_path / "current_state.npz"
        sim.export_snapshot(current_snap_path, source="current")
        print(f"✓ Exported current state to {current_snap_path.name}")
        
        # Export the named snapshot
        named_snap_path = tmp_path / "checkpoint_1.npz"
        sim.export_snapshot(named_snap_path, source="snapshot", name="checkpoint_1")
        print(f"✓ Exported named snapshot to {named_snap_path.name}")
        
        # Inspect the snapshots without loading them
        print("\n--- Snapshot Inspection ---")
        for path in [current_snap_path, named_snap_path]:
            meta = sim.inspect_snapshot(path)
            print(f"{path.name}:")
            print(f"  Schema: {meta['schema']}")
            print(f"  Name: {meta['name']}")
            print(f"  Time: {meta['t_curr']:.3f}")
            print(f"  State: x={meta['state_names'][0]}")
            print(f"  Created: {meta['created_at']}")
            print()
        
        # Continue simulation further
        sim.run(t_end=8.0, resume=True)
        current_x = sim._session_state.y_curr[0]
        print(f"✓ Continued to t=8.0, current state: x={current_x:.4f}")
        
        # Import the earlier snapshot
        print("\n--- Importing Snapshot ---")
        sim.import_snapshot(named_snap_path)
        restored_x = sim._session_state.y_curr[0]
        restored_t = sim._session_state.t_curr
        print(f"✓ Imported snapshot from t=2.0")
        print(f"  Restored state: t={restored_t:.3f}, x={restored_x:.4f}")
        
        # Verify results are cleared
        try:
            results = sim.raw_results()
            print("✗ ERROR: Results should be cleared after import!")
        except RuntimeError as e:
            print(f"✓ Results correctly cleared: {e}")
        
        # Can continue from restored state
        sim.run(t_end=3.0, resume=True)
        final_x = sim._session_state.y_curr[0]
        print(f"✓ Resumed from restored state to t=3.0, x={final_x:.4f}")
        
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()