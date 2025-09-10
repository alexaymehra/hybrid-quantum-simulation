"""
Filename: optimize_info.py
Author: Alexay Mehra
Date: 2025-09-09
Description: Builds the gate sequence, defines the cost function, creates instance of the time evolution to match
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.utils.constants import time, mass, diss_energy, width_param, equib_length, omega, chi, hbar, morse_cap, N
from src.utils.myops import D_full, R_full, H_On_Evo, MH_Evo

# Build the Gate Sequence (Always-On Time Evolution, Displacement Gate, XY Rotation Gate)
def gate_seq(params, d):
    """
    Builds the Gate Sequence

    Params: Each layer is a list of 4 parameters
    1. alpha_real - real part of alpha which shifts position
    2. alpha_imag - imaginary part of alpha which shifts momentum
    3. theta - one of the parameters for the xy qubit rotation gate
    4. phi - one of the parameters for the xy qubit rotaation gate
    
    """
    U = np.eye(2 * N, dtype=complex)
    for j in range(d):
        alpha_real = params[4*j]
        alpha_imag = params[4*j+1]
        theta = params[4*j+2]
        phi = params[4*j+3]
        
        D = D_full(alpha_real + 1j * alpha_imag)
        R = R_full(theta, phi)
        V = H_On_Evo(time)

        U = U @ V @ R @ D
    return U

# Definition of the cost (infidelity) function
def fidelity_loss(params, d, U_target):
    U = gate_seq(params, d)
    dim = U.shape[0]
    fid = np.abs(np.trace(U.conj().T @ U_target))/ (dim)
    return 1 - fid

# Create an Instance of the Target Morse Hamiltonian Time Evolution
morse_to_optimize = MH_Evo(t=time, u=mass, de=diss_energy, b=width_param, x0=equib_length)

