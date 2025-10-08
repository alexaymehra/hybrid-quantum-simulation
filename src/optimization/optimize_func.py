"""
Filename: optimize_func.py
Author: Alexay Mehra
Date: 2025-09-29
Description: Contains the optimization functions and a function to print optimal parameters
"""


# Imports
import numpy as np
import scipy.optimize as sp_opt


# Main Optimization Function -------------------------------------------------------------
# ----------------------------------------------------------------------------------------
def run_optimization(seq_template, backend, d, target_evolution, init_state, trace_index, loss_function,  mode='simple', max_iterations=5):
    """
    Main function to run optimizations
    Args:
        seq_template: template for one layer of gates
        backend: backend for the simulation
        d: layer depth
        target_evolution: target evolution to try to match
        init_state: initial qubit-qumode state
        trace_index: an `int` to indicate what component to keep from partial trace
        loss_function: cost function for the optimization
        mode: "simple" or "coordinate-descent"
        max_iterations: maximum number of iterations for coordinate descent optimization
    Returns:
        optimized paramaters of the gate sequence to match the target evolution
    """
    
    # total number of parameters across d layers
    n_params_per_layer = sum(g.n_params for g in seq_template)
    total_params = n_params_per_layer * d
    
    init_guess = np.random.rand(total_params) * 0.1   # small random initial guess
    
    if mode == 'coordinate-descent':
        return coordinate_descent_optimization(seq_template, backend, d, target_evolution, init_state, trace_index, loss_function, init_guess, max_iterations)
    else:
        return simple_optimization(seq_template, backend, d, target_evolution, init_state, trace_index, loss_function, init_guess)
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# Simple Optimization Function -----------------------------------------------------------
# ----------------------------------------------------------------------------------------
def simple_optimization(seq_template, backend, d, target_evolution, init_state, trace_index, loss_function, init_guess):
    result = sp_opt.minimize(
        loss_function,          
        init_guess,
        args=(seq_template, backend, d, target_evolution, init_state, trace_index),
        method='BFGS',
        options={'disp': True}
    )
    return result.x
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# Optimization that works on one gate type at a time -------------------------------------
# ----------------------------------------------------------------------------------------
def coordinate_descent_optimization(seq_template, backend, d, target_evolution, init_state, trace_index, loss_function, init_guess, max_iter):
    """
    Coordinate descent optimization, grouped by gate type across layers.
    All gates of the same type (e.g. all Displacements) are optimized together.
    """
    curr_params = init_guess.copy()                 

    # --- Build groups by gate type across all layers ---
    groups = {}
    idx = 0
    for layer in range(d):
        for gate in seq_template:
            if gate.n_params == 0:
                continue
            indices = list(range(idx, idx + gate.n_params))
            groups.setdefault(gate.name, []).extend(indices)
            idx += gate.n_params

    # --- Optimization loop ---
    for it in range(max_iter):
        for gname, indices in groups.items():

            # cost function restricted to a single gate type
            # only positions in that block's indicies will be changed
            def f_block(block_vars):
                temp = curr_params.copy()
                temp[indices] = block_vars
                return loss_function(temp, seq_template, backend, d, target_evolution, init_state, trace_index)

            # current values for this blocks
            x0_block = curr_params[indices]

            res = sp_opt.minimize(
                f_block,
                x0_block,
                method="BFGS",
                options={"disp": False}
            )

            # modifies the parameters in place so next iteration sees updated values
            curr_params[indices] = res.x
            
            # print optimization information
            curr_infid = loss_function(curr_params, seq_template, backend, d, target_evolution, init_state, trace_index)
            print(f"Optimized {gname}, Iteration {it + 1}")
            print(f"Current Infidelity: {curr_infid:.6f}\n")

    return curr_params
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# Print the optimal parameters found -----------------------------------------------------
# ----------------------------------------------------------------------------------------
def print_optimal_params(params, seq_template, d):
    print("Optimized Parameters")
    idx = 0
    for layer in range(d):
        print(f"Layer {layer+1}:")
        for gate in seq_template:
            gate_params = params[idx: idx + gate.n_params]
            print(f"  {gate.name}: {gate_params}")
            idx += gate.n_params
        print("-" * 30)
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------