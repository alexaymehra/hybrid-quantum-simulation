"""
Filename: optimize_func.py
Author: Alexay Mehra
Date: 2025-09-08
Description: Contains the optimization function and function to print optimal parameters
"""

# Imports
import numpy as np
import scipy as sp

from optimize_info import fidelity_loss, morse_to_optimize


# Optimization Function
def run_optimization(d):
    num_params = d * 4      # [Re(α), Im(α), θ, φ] per gate
    init_guess = np.random.rand(num_params) * 0.1   # Provides small initial guess near 0

    result = sp.optimize.minimize(
        fidelity_loss,          
        init_guess,
        args=(d, morse_to_optimize),
        method='BFGS',
        options={'disp': True}
    )

    return result

# Function to Print the Optimized Parameters
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
