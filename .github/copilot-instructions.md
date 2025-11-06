# Copilot Instructions for MorsePotential

## Project Overview
This repository simulates molecular interactions using quantum harmonic oscillators. It features optimization pipelines (NumPy/SciPy), quantum simulation utilities (QuTip), and visualization tools (Matplotlib). The codebase is organized for modularity and experimentation.

## Architecture & Key Components
- `src/optimization/`: Contains routines for calibrating quantum circuit models. Entry point: `optimize_func.py`.
- `src/simulation/`: Implements simulation logic for quantum systems. Main file: `simulate.py`.
- `src/utils/`: Provides helper functions, gate specifications, and Morse potential calculations. Key files: `gates.py`, `morse.py`, `gate_specs.py`.
- `src/visualization/`: Handles plotting and visualization, especially Wigner functions (`wigner.py`).
- `notebooks/`: Jupyter notebooks for prototyping, experiments, and system tests.
- `tests/`: Contains notebook-based tests for system and visualization validation.

## Developer Workflows
- **Run experiments:** Use notebooks in `notebooks/` for interactive development and prototyping.
- **Testing:** Tests are implemented as Jupyter notebooks in `tests/`. Run and validate by executing all cells.
- **Dependencies:** Install with `pip install -r requirements.txt`. Core dependencies: NumPy, SciPy, QuTip, Matplotlib.
- **Debugging:** Use print statements or notebook cell outputs for stepwise inspection. No custom logging framework.

## Project-Specific Patterns
- **Parameter Passing:** Functions and classes expect explicit parameter dictionaries for circuit and simulation settings.
- **Modularity:** Utilities and routines are split by domain (optimization, simulation, visualization, utils). Import from `src` submodules.
- **Notebook-Driven Development:** Most workflows and tests are in notebooks, not standalone scripts.
- **Visualization:** Wigner function plots are generated via `visualization/wigner.py` and used in notebooks.

## Integration Points
- **QuTip:** Used for quantum simulation. See `src/simulation/simulate.py` and `src/utils/gates.py` for integration patterns.
- **Matplotlib:** Used for all visualizations. Plots are typically generated in notebooks or via `visualization/` scripts.

## Conventions
- **No main script:** Entry points are notebooks or specific module functions.
- **Explicit imports:** Avoid wildcard imports; use full module paths.
- **Testing in notebooks:** Validate changes by running notebook-based tests, not via pytest/unittest.

## Example: Adding a New Gate
- Implement gate logic in `src/utils/gates.py`.
- Update `gate_specs.py` for parameter definitions.
- Use/test new gate in a notebook in `notebooks/`.

---
For questions or unclear conventions, review the README or ask for clarification. Update this file as new patterns emerge.
