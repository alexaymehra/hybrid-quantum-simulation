"""
Filename: optimize_func.py
Author: Alexay Mehra
Date: 2025-09-12
Description: Contains the optimization functions and a function to print optimal parameters
"""

# Imports
import numpy as np
import scipy.optimize as sp_opt

from .optimize_info import fidelity_loss, morse_to_optimize


# Main Optimization Function
def run_optimization(d, mode='simple', max_iterations=5):
    # there will be a [Re(α), Im(α), θ, φ] per gate sequence
    init_guess = np.random.rand(d * 4) * 0.1   # Provides small initial guess near 0

    if (mode == 'coordinate-descent'):
        return coordinate_descent_optimization(d, init_guess, max_iterations)
    else:
        return simple_optimization(d, init_guess)
    


def simple_optimization(d, init_guess):
    result = sp_opt.minimize(
        fidelity_loss,          
        init_guess,
        args=(d, morse_to_optimize),
        method='BFGS',
        options={'disp': True}
    )

    return result.x



def coordinate_descent_optimization(d, init_guess, max_iter):
    # Optimizes Parameters one at a time (coordinate descent)

    curr_params = init_guess.copy()

    # Dictionary to hold group names and indices
    groups = {
        "alpha_real": np.arange(0, d * 4, 4),
        "alpha_imag": np.arange(1, d * 4, 4),
        "theta":      np.arange(2, d * 4, 4),
        "phi":        np.arange(3, d * 4, 4),
    }

    # number of loops through all blocks
    for it in range(max_iter):

        # passes through each block in turn
        for group_name, indices in groups.items():
            
            # const function restricted to just one block
            # only the positions indicated by indices will be replaced by candidate values block_vars
            # since only the chosen block is changed, the cost is optimized with respect to just those vars
            def f_block(block_vars):
                temp = curr_params.copy()
                temp[indices] = block_vars
                return fidelity_loss(temp, d, morse_to_optimize)
            
            # Current values for this block
            x0_block = curr_params[indices]
            
            res = sp_opt.minimize(
                f_block,
                x0_block,
                method="BFGS",
                options={"disp": False}
            )

            curr_params[indices] = res.x # init_guess gets modified in place so the next iteration sees updated values
            curr_infid = fidelity_loss(curr_params, d, morse_to_optimize)
            print(f"Optimized {group_name}, Iteration {it}\nCurrent Infidelity: {curr_infid}\n")
    
    return curr_params



def print_optimal_params(params, d):
    print("Optimized Paramters")
    for i in range(d):
        re_alpha = params[i * 4 + 0]
        im_alpha = params[i * 4 + 1]
        theta    = params[i * 4 + 2]
        phi      = params[i * 4 + 3]
        
        print(f"Gate {i+1}:")
        print(f"  α     = {re_alpha:.4f} + {im_alpha:.4f}j")
        print(f"  θ     = {theta:.4f}")
        print(f"  φ     = {phi:.4f}")
        print("-" * 30)
