"""
Filename: myops.py
Author: Alexay Mehra
Date: 2025-09-08
Description: Builds operators that will be used in the simulation:
- Morse Hamiltonian and time evolution
- Always-on Hamiltonain
- Displacement gate
- Qubit xy-rotation gate
- Other prerequisite gates
"""

# Imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.utils.constants import time, mass, diss_energy, width_param, equib_length, omega, chi, hbar, morse_cap, N


I_q = np.eye(2) #Identity operator for qubits
I_o = np.eye(N) #Identity operator for qumodes

a = np.diag(np.sqrt(np.arange(1, N)), 1)  # Annihilation operator
adag = a.T.conj()                         # Creation operator
n_op = adag @ a                           # Photon number operator

# Pauli Matrices for qubit
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

H_always_on = (chi * sigma_z + omega * np.eye(2))  # Always-on Hamiltonian for a cavity-QED System
H_always_on = np.kron(H_always_on, n_op)           # Full Always-on Hamiltonian (Qubit-Qumode Hilbert Space)

# Always-on Time Evolution
def H_On_Evo(t):
    return sp.linalg.expm(-1j * H_always_on * t)

# Dispacement Gate 
def displacement(alpha):
    A = alpha * adag - np.conj(alpha) * a
    return sp.linalg.expm(A)

# Dispacement Gate (Qubit-Qumode Hilbert Space)
def D_full(alpha):
    return np.kron(I_q, displacement(alpha))

# Qubit XY Rotation Gate
def R_phi(theta, phi):
    op = (np.cos(phi) * sigma_x + np.sin(phi) * sigma_y)
    return sp.linalg.expm(-1j * theta / 2 * op)

# Qubit XY Rotation Gate (Qubit-Qumode Hilbert Space)
def R_full(theta, phi):
    return np.kron(R_phi(theta, phi), I_o)


# Morse Hamiltonian (Returns the Matrix from the Qubit-Qumode Hilbert Space)
def H_Morse(u ,de, b, x0):
    """
    Morse Hamiltonian Gate

    Args:
        u (real): reduced mass of diatomic system
        de (real): dissociation energy
        b (real): width parameter
        x0 (real): equilibrium bond length

    Returns:
        csc_matrix: operator matrix
    """
    m_omega = np.sqrt(2 * de * (b ** 2) / u)
    X_op = (a + adag) / np.sqrt(2)
    P_op = 1j * (a - adag) / (np.sqrt(2))
    x_op = X_op * np.sqrt(hbar / (u * m_omega))
    p_op = P_op * np.sqrt(hbar * u * omega)

    kin_op = (p_op @ p_op) / (2 * u)

    exp_term = sp.linalg.expm(-1 * b * (x_op - x0 * np.eye(N)))
    mp_op = de * ((np.eye(N) - exp_term) @ (np.eye(N) - exp_term))

    full_m = kin_op + mp_op

    full_m = np.kron(I_q, full_m)

    return full_m

# Time Evolution for the Morse Hamiltonian
def MH_Evo(t, u, de, b, x0):
    return sp.linalg.expm(-1j * H_Morse(u, de, b, x0) * t)
