"""
Filename: gates.py
Author: Alexay Mehra
Date: 2025-10-16
Description: Matrix definitions for relevant gates.
"""


# Imports
import numpy as np
import scipy as sp
import math


# Gates Class ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
class Gates:
    """Builds operators for qubit-qumode systems"""

    I_q = np.eye(2)                                         # Identity operator for qubits

    proj0 = np.array([[1, 0], [0, 0]])                      # Projector to qubit 0 state
    proj1 = np.array([[0, 0],[0, 1]])                       # Projector to qubit 1 state

    sigma_x = np.array([[0, 1], [1, 0]])                    # Pauli X Gate
    sigma_y = np.array([[0, -1j], [1j, 0]])                 # Pauli Y Gate
    sigma_z = np.array([[1, 0], [0, -1]])                   # Pauli Z Gate
    

    def __init__(self, dim=20):
        """Specify qumode dimension
       
        Args:
            dim: dimension of the qumode Hilbert space
        """
        self.dim = dim

        self.I_o = np.eye(dim)                              # Identity operator for qumodes

        self.a = np.diag(np.sqrt(np.arange(1, dim)), 1)     # Annihilation operator
        self.adag = np.transpose(np.conj(self.a))           # Creation operator
        self.n_op = self.adag @ self.a                      # Photon number operator

        self.x_op = (self.a + self.adag)/(np.sqrt(2))       # Position quadrature operator
        self.p_op = -1j * (self.a - self.adag)/(np.sqrt(2)) # Momentum quadrature operator


    def c_r_ham(self, omega=0.0, chi=0.5):
        """Hamiltonain for the controlled rotation gate (qubit-qumode space)

        Args:
            omega: cavity frequency
            chi: dispersive coupling

        Returns:
            `ndarray`: operator matrix
        """
        qubit_part = (chi * self.sigma_z + omega * self.I_q)      
        hamiltonian = np.kron(qubit_part, self.n_op)
        return hamiltonian
    
    def c_r(self, omega=0.0, chi=0.5, t=1.0):
        """Controlled rotation gate (qubit-qumode space)
        
        Args:
            omega: cavity frequency
            chi: dispersive coupling
            t: evolution time (this should be morse evolution time / sequence depth)
        
        Returns:
            `ndarray`: operator matrix
        """
        evolution = sp.linalg.expm(-1j * self.c_r_ham(omega, chi) * t)
        return evolution

    def cv_r(self, theta):
        """Phase space rotation gate (qumode space)

        Args:
            theta: rotation angle
        
        Returns:
            `ndarray`: operator matrix
        """
        gate = sp.linalg.expm(-1j * theta * self.n_op)
        return gate


    def cv_r_full(self, theta):
        """Phase space rotation gate (qubit-qumode space)
        
        Args:
            alpha: rotation angle
        
        Returns:
            `ndarray`: operator matrix
        """
        full_gate = np.kron(self.I_q, self.cv_r(theta))
        return full_gate


    def cv_d(self, alpha):
        """Displacement gate (qumode space)
    
        Args:
            alpha: displacement amount
        
        Returns:
            `ndarray`: operator matrix
        """
        exponent = (alpha * self.adag) - ((np.conj(alpha)) * self.a)
        gate = sp.linalg.expm(exponent)
        return gate
    

    def cv_d_full(self, alpha):
        """Displacement gate (qubit-qumode space)
        
        Args:
            alpha: displacement amount
        
        Returns:
            `ndarray`: operator matrix`
        """
        full_gate = np.kron(self.I_q, self.cv_d(alpha))
        return full_gate
    

    def cv_s(self, theta):
        """Squeezing gate (qumode space)
        
        Args:
            theta: squeezing strength

        Returns:
            `ndarray`: operator matrix
        """
        a_part = 0.5 * (self.a @ self.a - self.adag @ self.adag)
        gate = sp.linalg.expm(theta * a_part)
        return gate


    def cv_s_full(self, theta):
        """Squeezing gate (qubit-qumode space)

        Args:
            theta: squeezing strength

        Returns:
            `ndarray`: operator matrix
        """
        full_gate = np.kron(self.I_q, self.cv_s(theta))
        return full_gate


    def c_d(self, alpha):
        """Controlled displacement gate (qubit-qumode space)
        
        Args:
            alpha: displacement amount

        Returns:
            =`ndarray`: operator matrix
        """
        exponent = alpha * self.adag - np.conj(alpha) * self.a
        disp = sp.linalg.expm(exponent)
        part0 = np.kron(self.proj0, self.I_o)
        part1 = np.kron(self.proj1, disp)
        gate = part0 + part1
        return gate

    def q_xy_r(self, theta, phi):
        """Qubit xy rotation gate (qubit space)
        
        Args:
            theta: rotation angle
            phi: rotation axis

        Returns:
            `ndarray`: operator matrix
        """
        exponent = (np.cos(phi) * self.sigma_x + np.sin(phi) * self.sigma_z)
        gate = sp.linalg.expm(-1j * (theta * 2) * exponent)
        return gate

    def q_xy_r_full(self, theta, phi):
        """Qubit xy rotation gate (qubit-qumode space)
        
        Args:
            theta: rotation angle
            phi: rotation axis

        Returns:
            `ndarray`: operator matrix
        """
        full_gate = np.kron(self.q_xy_r(theta, phi), self.I_o)
        return full_gate

    def m_ham(self, mp):
        """Morse Hamiltonian (qumode space)
        
        Args:
            mp: `MorsePotential` object
        
        Returns:
            `ndarray`: operator matrix
        """
        kinetic_part = (self.p_op @ self.p_op) 
        exponent_term = -1 * mp.b * (self.x_op - mp.x0 * self.I_o)
        position_part = mp.de * ((self.I_o - sp.linalg.expm(exponent_term)) @ (self.I_o - sp.linalg.expm(exponent_term)))
        hamiltonian = kinetic_part + position_part

        return hamiltonian
    
    def m_ham_full(self, mp):
        """Morse Hamiltonian (qubit-qumode space)

        Args:
            mp: `MorsePotential` object

        Returns:
            `ndarray`: operator matrix
        """
        full_hamiltonian = np.kron(self.I_q, self.m_ham(mp))
        return full_hamiltonian
    
    def m_ham_expansion(self, mp, n):
        """Morse Hamiltonian expansion (qumode space)
        
        Args:
            mp: `MorsePotential` object
            n: expansion order (minimum 2)

        Returns:
            `ndarray`: operator matrix
        """
        if n < 2:
            raise Exception("Expansion order must be at least 2")

        kinetic_part = self.p_op @ self.p_op
        hamiltonian = kinetic_part

        # Compute the coefficients for the expansion        
        a_k = {}
        for k in range(1, n):
            a_k[k] = (-1) ** (k + 1) / math.factorial(k)

        # Compute the cauchy product coefficients
        c_n = {}
        for i in range(2, n+1):
            c_n[i] = sum(a_k[m] * a_k[i-m] for m in range(1, i))
        
        # Matrix for expansion terms
        single_x = self.x_op - mp.x0 * self.I_o

        # Build the full expansion
        for l in range(2, n+1):
            curr_term = np.linalg.matrix_power(single_x, l)
            hamiltonian += mp.de * c_n[l] * (mp.b ** l) * curr_term

        return hamiltonian

    def m_ham_expansion_full(self, mp, n):
        """Morse Hamiltonian expansion (qubit-qumode space)

        Args:
            mp: `MorsePotential` object
            n: expansion order (minimum 2)

        Returns:
            `ndarray`: operator matrix
        """
        full_hamiltonian = np.kron(self.I_q, self.m_ham_expansion(mp, n))
        return full_hamiltonian

    def m_evo(self, mp, t):
        """Morse time evolution (qumode space)
        
        Args:
            mp: `MorsePotential` object
            t: evolution time

        Returns:
            `ndarray`: operator matrix
        """
        time_evolution = sp.linalg.expm(-1j * self.m_ham(mp) * t)
        return time_evolution
    
    def m_evo_expansion(self, mp, n, t):
        """Morse time evolution expansion (qumode space)
       
        Args:
            mp: `MorsePotential` object
            n: expansion order (minimum 2)
            t: evolution time

        Returns:
            `ndarray`: operator matrix
        """
        if n < 2:
            raise Exception("Expansion order must be at least 2")

        time_evolution = sp.linalg.expm(-1j * self.m_ham_expansion(mp, n) * t)
        return time_evolution

    def m_evo_full(self, mp, t):
        """Morse time evolution (qubit-qumode space)
        
        Args:
            mp: `MorsePotential` object
            t: evolution time

        Returns:
            `ndarray`: operator matrix
        """
        full_time_evolution = np.kron(self.I_q, self.m_evo(mp, t))
        return full_time_evolution

    def m_evo_expansion_full(self, mp, n, t):
        """Morse time evolution expansion (qubit-qumode space)

        Args:
            mp: `MorsePotential` object
            n: expansion order (minimum 2)
            t: evolution time

        Returns:
            `ndarray`: operator matrix
        """
        full_time_evolution = np.kron(self.I_q, self.m_evo_expansion(mp, n, t))
        return full_time_evolution
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------