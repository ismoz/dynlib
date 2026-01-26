# examples/print_equations_demo.py
"""
Demonstrate printing model equations from the DSL spec.

Prints the equations for the built-in Henon map and Lorenz system.
"""

from dynlib import setup


def main() -> None:
    print("Henon map equations:")
    sim_henon = setup("builtin://map/henon", stepper="map", jit=False)
    sim_henon.model.print_equations()

    print("\nLorenz system equations:")
    sim_lorenz = setup("builtin://ode/lorenz", stepper="rk4", jit=False)
    sim_lorenz.model.print_equations(tables=("equations", "equations.jacobian"))

if __name__ == "__main__":
    main()
