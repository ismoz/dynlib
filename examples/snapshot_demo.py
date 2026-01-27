# examples/snapshot_demo.py
"""
Demonstration of snapshot export/import functionality in dynlib.

This example shows how to:
1. Export current simulation state to disk (including workspaces)
2. Export named snapshots to disk
3. Import snapshots from disk
4. Inspect snapshot metadata
5. Demonstrate workspace persistence across sessions
"""

from pathlib import Path
import tempfile
import tomllib
from dynlib import setup


def create_model():
    """Create a simple exponential decay model for demonstration."""
    model_uri = """
inline:
[model]
type = "ode"
name = "Demo Exponential Decay"
dtype = "float64"

[states]
x = 1.0

[params]
decay_rate = 0.1

[equations.rhs]
x = "-decay_rate * x"

[sim]
stepper = "euler"
dt = 0.01
t0 = 0.0
t_end = 10.0
record = true
"""

    return setup(model_uri, jit=True)
def main():
    print("=== Dynlib Snapshot Export/Import Demo ===\n")

    # Create simulation
    sim = create_model()
    print("✓ Created simulation with exponential decay model")

    # Run to some intermediate state
    sim.run(T=2.0)
    print(f"✓ Ran simulation to t=2.0, current state: x={sim.state_vector()[0]:.4f}")

    # Create an in-memory snapshot
    sim.create_snapshot("checkpoint_1", "After running to t=2.0")
    print("✓ Created in-memory snapshot 'checkpoint_1'")

    # Continue running
    sim.run(T=5.0, resume=True)
    print(f"✓ Continued to t=5.0, current state: x={sim.state_vector()[0]:.4f}")

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
            print(f"  Workspace signature: {meta['pins']['workspace_sig']}")
            print(f"  Created: {meta['created_at']}")
            print()

        # Continue simulation further
        sim.run(T=8.0, resume=True)
        current_x = sim.state_vector()[0]
        print(f"✓ Continued to t=8.0, current state: x={current_x:.4f}")

        # Import the earlier snapshot
        print("\n--- Importing Snapshot ---")
        sim.import_snapshot(named_snap_path)
        restored_x = sim.state_vector()[0]
        restored_t = sim.session_state_summary()["t"]
        print(f"✓ Imported snapshot from t=2.0")
        print(f"  Restored state: t={restored_t:.3f}, x={restored_x:.4f}")

        # Verify results are cleared
        try:
            results = sim.raw_results()
            print("✗ ERROR: Results should be cleared after import!")
        except RuntimeError as e:
            print(f"✓ Results correctly cleared: {e}")

        # Can continue from restored state
        sim.run(T=3.0, resume=True)
        final_x = sim.state_vector()[0]
        print(f"✓ Resumed from restored state to t=3.0, x={final_x:.4f}")

        # Demonstrate workspace persistence
        print("\n--- Workspace Persistence ---")
        # Create a new simulation to show workspace restoration
        sim2 = create_model()
        sim2.run(T=1.0)  # Run to different state

        # Export current state with workspace
        ws_snap_path = tmp_path / "workspace_demo.npz"
        sim.export_snapshot(ws_snap_path, source="current")
        print("✓ Exported simulation state with workspace")

        # Import into new simulation
        sim2.import_snapshot(ws_snap_path)
        print("✓ Imported state with workspace into new simulation")

        # Verify state was restored
        restored_state = sim2.session_state_summary()
        print(f"  Restored time: {restored_state['t']:.3f}")
        print(f"  Restored state: x={sim2.state_vector()[0]:.4f}")
        print(f"  Can resume: {restored_state['can_resume']}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
