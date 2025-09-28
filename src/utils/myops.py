"""
Filename: myops.py
Author: Alexay Mehra
Date: 2025-09-28

"""

# Imports
import numpy as np
import scipy as sp


class Gates():
    dim = 20
    time = 1

    I_q = np.eye(2)                                     # Identity operator for qubits
    I_o = np.eye(dim)                                   # Identity operator for qumodes

    a = np.diag(np.sqrt(np.arange(1, dim)), 1)          # Annihilation operator
    adag = np.transpose(np.conj(a))                     # Creation operator
    n_op = adag @ a                                     # Photon number operator

    x_op = (a + adag)/(np.sqrt(2))                      # Position quadrature operator
    p_op = -1j * (a - adag)/(np.sqrt(2))                # Momentum quadrature operator

    sigma_x = np.array([[0, 1], [1, 0]])                # Pauli X Gate
    sigma_y = np.array([[0, -1j], [1j, 0]])             # Pauli Y Gate
    sigma_z = np.array([[1, 0], [0, -1]])               # Pauli Z Gatez

    def __init__(self, dim=20, time=1):
        self.dim = dim
        self.time = time
        return
    
    def always_on_evolution(self, omega=1, chi=0.01, t=time):
        qubit_part = (chi * self.sigma_z + omega * self.I_q)      
        hamiltonian = np.kron(qubit_part, self.n_op)
        evolution = sp.linalg.expm(-1j * hamiltonian * t)
        return evolution
    
    def displacement(self, alpha):
        exponent = alpha * self.adag - np.conj(alpha) * self.a
        gate = sp.linalg.expm(exponent)
        return gate
    
    def full_displacement(self, alpha):
        full_gate = np.kron(self.I_q, self.displacement(alpha))
        return full_gate

    def qubit_xy_rotation(self, theta, phi):
        exponent = (np.cos(phi) * self.sigma_x + np.sin(phi) * self.sigma_z)
        gate = sp.linalg.expm(-1j * (theta * 2) * exponent)
        return gate

    def full_qubit_xy_rotation(self, theta, phi):
        full_gate = np.kron(self.qubit_xy_rotation(theta, phi), self.I_o)
        return full_gate

    def morse_hamiltonian(self, mp):
        """
        Create the Morse Hamiltonian from the given parameters
        """
        phys_x = np.sqrt(mp.hbar / (mp.mass * mp.angular_freq)) * self.x_op
        phys_p = np.sqrt(mp.hbar * mp.mass * mp.angular_freq) * self.p_op
        
        kinetic_part = (phys_p @ phys_p) * (1 / (2 * mp.mass))
        exponent_term = -1 * mp.b * (phys_x - mp.x0 * self.I_o)
        position_part = mp.de * ((1 - sp.linalg.expm(exponent_term)) ** 2)
        hamiltonian = kinetic_part + position_part

        full_kin_part = np.kron(self.I_q, kinetic_part)
        full_pos_part = np.kron(self.I_q, position_part)
        full_hamiltonian = np.kron(self.I_q, hamiltonian)

        return full_hamiltonian, full_pos_part, full_kin_part
    
    def morse_time_evolution(self, mp):
        ham, pos, kin = self.morse_hamiltonian(mp)
        exp_ham = sp.linalg.expm(-1j* ham * self.time)
        exp_pos = sp.linalg.expm(-1j* pos * self.time)
        exp_kin = sp.linalg.expm(-1j* kin * self.time)
        return exp_ham, exp_pos, exp_kin