import numpy as np
import warnings
from dynlib import setup, list_steppers, get_stepper

# Suppress RuntimeWarnings to avoid cluttering output with solver convergence messages
warnings.simplefilter("ignore", RuntimeWarning)

# Define models and their exact solutions
models = [
    {
        "name": "Exponential Decay",
        "model": '''
inline:
[model]
type="ode"

[states]
x=1.0

[params]
a=1.0

[equations]
expr = """
dx = -a * x
"""
''',
        "exact": lambda t: np.exp(-t)
    },
    {
        "name": "Harmonic Oscillator",
        "model": '''
inline:
[model]
type="ode"

[states]
x=1.0
v=0.0

[params]
omega=100.0

[equations]
expr = """
dx = v
dv = -omega**2 * x
"""
''',
        "exact": lambda t: (np.cos(100.0 * t), -100.0 * np.sin(100.0 * t))
    }
]

# Simulation parameters
T = 10  # total time
dt = 1e-4  # time step
N = int(T / dt)  # number of steps

ode_steppers = list_steppers(kind="ode")

for model_info in models:
    model_name = model_info["name"]
    model_str = model_info["model"]
    exact_func = model_info["exact"]
    
    print(f"\n=== {model_name} ===")
    
    # Dictionary to store errors
    errors = {}
    failed_steppers = {}
    
    for name in ode_steppers:
        spec = get_stepper(name)
        try:        
            print(f"Running simulation with stepper: {name}")
            
            # Setup simulation
            sim = setup(model_str, stepper=name, jit=False)
            sim.config(dt=dt, max_steps=N*10)  # allow more steps if needed
            
            # Run simulation
            sim.run(T=T)
            res = sim.results()
            
            # Compute relative error on x
            if model_name == "Exponential Decay":
                x_exact = exact_func(res.t)
                rel_error = np.abs((res["x"] - x_exact) / x_exact)
            else:  # Harmonic Oscillator
                x_exact, v_exact = exact_func(res.t)
                rel_error = np.abs((res["x"] - x_exact) / x_exact)
            
            rms_rel_error = np.sqrt(np.mean(rel_error**2))
            
            errors[name] = rms_rel_error
            print(f"RMS relative error for {name}: {rms_rel_error:.2e}")
            
        except Exception as e:
            print(f"Error with stepper {name}: {e}")
            failed_steppers[name] = str(e)
            continue
    
    # Sort steppers by accuracy (lowest error first)
    sorted_steppers = sorted(errors.items(), key=lambda x: x[1])
    
    print(f"\n{model_name} - Stepper accuracy ranking (lowest RMS relative error = most accurate):")
    for rank, (name, error) in enumerate(sorted_steppers, 1):
        print(f"{rank}. {name}: {error:.2e}")
    
    if failed_steppers:
        print(f"\n{model_name} - Failed steppers:")
        for name, error_msg in failed_steppers.items():
            print(f"- {name}: {error_msg}")