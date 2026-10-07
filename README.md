# MTH321-TEAM9

## Robertson Stiff Chemical Kinetics Problem

This project studies the Robertson chemical kinetics problem, a classic stiff system of ordinary differential equations. The three chemical species change at very different speeds. This creates both fast and slow time scales, which makes the problem challenging for numerical methods.

The project compares explicit and implicit methods and studies how stiffness affects stability, accuracy, step size, and physical results. It also checks conservation, non-negativity, and error against a high-accuracy reference solution.

## Run

### Check the Python environment

Run:

```bash
python setup.py
```

This checks whether Python, NumPy, SciPy, and Matplotlib are available and prints their version numbers. If a required package is missing, the script reports which package needs to be installed.

### Run the experiments

Run:

```bash
python experiments.py
```

This script runs the main experiments. It calculates reference solutions with Radau and BDF, studies the Jacobian eigenvalues and stiffness ratio, tests explicit Euler with different step sizes, and checks conservation, negative concentrations, and full-state error.

It also measures convergence orders for Euler, Heun, RK4, backward Euler, and the trapezoidal method; tests adaptive step-size control; compares the final value of `y2` at `t = 40`; studies L-stability; and records Newton iteration residuals.

The script saves numerical tables as CSV files and figures as PNG files.

### Plot the stability regions

Run:

```bash
python stability_regions.py
```

This script plots the absolute-stability regions of explicit Euler, Heun, RK4, backward Euler, and the trapezoidal method. It also uses the Robertson Jacobian eigenvalues at `t = 40` to estimate step-size stability limits for the explicit methods.

## Project files

- `robertson.py`: Defines the Robertson ODE model, its analytic Jacobians, and the numerical methods: explicit Euler, Heun, RK4, backward Euler, and the trapezoidal method. It also contains the damped Newton solver, adaptive step-size controller, reference-solution function, output grids, and error and conservation diagnostics.
- `experiments.py`: Runs the numerical experiments using functions from `robertson.py`. It generates figures and CSV tables for analysing the methods.
- `stability_regions.py`: Calculates and plots the methods’ absolute-stability regions and estimates explicit-method step-size limits using the Robertson eigenvalues.
- `setup.py`: Checks whether the required Python packages are installed and prints their version numbers.
- `requirements.txt`: Lists the Python packages needed to run the project.
- `figures/`: Stores generated plots.
- `results/`: Stores numerical results in CSV format.

Note: Check the output paths in `experiments.py`. In the current version, the script saves figures and results to folders one level above the project directory (`../figures` and `../results`). Change these paths if you want the output saved inside this repository’s `figures/` and `results/` folders.