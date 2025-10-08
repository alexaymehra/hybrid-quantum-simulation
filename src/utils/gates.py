"""
Filename: gates.py
Author: Alexay Mehra
Date: 2025-09-28
Description: Holds the matrix definitions for relevant gates.
"""


# Imports
import numpy as np
import scipy as sp


# Gates Class ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
class Gates:
    
    I_q = np.eye(2)                                         # Identity operator for qubits

    proj0 = np.array([[1, 0], [0, 0]])                      # Projector to qubit 0 state
    proj1 = np.array([[0, 0],[0, 1]])                       # Projector to qubit 1 state

    sigma_x = np.array([[0, 1], [1, 0]])                    # Pauli X Gate
    sigma_y = np.array([[0, -1j], [1j, 0]])                 # Pauli Y Gate
    sigma_z = np.array([[1, 0], [0, -1]])                   # Pauli Z Gatez
    

    def __init__(self, dim=20, time=1):
        self.dim = dim
        self.time = time

        self.I_o = np.eye(dim)                              # Identity operator for qumodes

        self.a = np.diag(np.sqrt(np.arange(1, dim)), 1)     # Annihilation operator
        self.adag = np.transpose(np.conj(self.a))           # Creation operator
        self.n_op = self.adag @ self.a                      # Photon number operator

        self.x_op = (self.a + self.adag)/(np.sqrt(2))       # Position quadrature operator
        self.p_op = -1j * (self.a - self.adag)/(np.sqrt(2)) # Momentum quadrature operator


    # The time used here should be the time the morse is optimized over divided by num steps
    def always_on_evolution(self, omega=1, chi=0.01, t=1):
        """
        Generate the Always On Evolution in the Qubit-Qumode Basis
        Args:
            omega: natural frequency of the harmonic oscillator
            chi: frequency shift per photon
            t: time to evolve over
        Returns:
            evolution: ndarray of shape (2 * qumode_dim, 2 * qumode_dim)
        """
        qubit_part = (chi * self.sigma_z + omega * self.I_q)      
        hamiltonian = np.kron(qubit_part, self.n_op)
        evolution = sp.linalg.expm(-1j * hamiltonian * t)
        return evolution
    

    def phase_space_rotation(self, theta):
        """
        Generate the Phase Space Rotation Gate in the Qumode Basis
        Args:
            theta: rotation angle
        Returns:
            gate: ndarray of shape (qumode_dim, qumode_dim)
        """
        gate = sp.linalg.expm(1j * theta * self.n_op)
        return gate


    def full_phase_space_rotation(self, theta):
        """
        Generate the Phase Space Rotation Gate in the Qubit-Qumode Basis
        Args:
            alpha: rotation angle
        Returns:
            gate: ndarray of shape (2 * qumode_dim, 2 * qumode_dim)
        """
        full_gate = np.kron(self.I_q, self.phase_space_rotation(theta))
        return full_gate


    def displacement(self, alpha):
        """
        Generate the Displacement Gate in the Qumode Basis
        Args:
            alpha: displacement amount in phase space
        Returns:
            gate: ndarray of shape (qumode_dim, qumode_dim)
        """
        exponent = alpha * self.adag - np.conj(alpha) * self.a
        gate = sp.linalg.expm(exponent)
        return gate
    

    def full_displacement(self, alpha):
        """
        Generate the Displacement Gate in the Qubit-Qumode Basis
        Args:
            alpha: displacement amount in phase space
        Returns:
            full_gate: ndarray of shape (2 * qumode_dim, 2 * qumode_dim)
        """
        full_gate = np.kron(self.I_q, self.displacement(alpha))
        return full_gate
    

    def squeezing(self, theta):
        """
        Generate the Squeezing Gate in the Qumode Basis
        Args:
            theta: squeezing strength
        Returns:
            gate: ndarray of shape (qumode_dim, qumode_dim)
        """
        a_part = 0.5 * (self.a @ self.a - self.adag @ self.adag)
        gate = sp.linalg.expm(theta * a_part)
        return gate


    def full_squeezing(self, theta):
        """
        Generate the Squeezing Gate in the Qubit-Qumode Basis
        Args:
            theta: squeezing strength
        Returns:
            full_gate: ndarray of shape (2 * qumode_dim, 2 * qumode_dim)
        """
        full_gate = np.kron(self.I_q, self.squeezing(theta))
        return full_gate


    def controlled_displacement(self, alpha):
        """
        Generate the Controlled Displacement Gate in the Qubit-Qumode Basis
        Args:
            alpha: displacement amount in phase space
        Returns:
            gate: ndarray of shape (2 * qumode_dim, 2 * qumode_dim)
        """
        exponent = alpha * self.adag - np.conj(alpha) * self.a
        disp = sp.linalg.expm(exponent)
        part0 = np.kron(self.proj0, self.I_o)
        part1 = np.kron(self.proj1, disp)
        gate = part0 + part1
        return gate


    def controlled_cv_rotation(self, theta):
        """
        Generate the Controlled CV Rotation Gate in the Qubit-Qumode Basis
        Args:
            theta: rotation angle
        Returns:
            gate: ndarray of shape (2 * qumode_dim, 2 * qumode_dim)
        """
        cv_rot = sp.linalg.expm(-1j * theta * self.n_op)
        part0 = np.kron(self.proj0, self.I_o)
        part1 = np.kron(self.proj1, cv_rot)
        gate = part0 + part1
        return gate


    def qubit_xy_rotation(self, theta, phi):
        """
        Generate the Qubit XY Rotation Gate in the Qubit Basis
        Args:
            theta: rotation angle
            phi: rotation axis
        Returns:
            gate: ndarray of shape (2, 2)
        """
        exponent = (np.cos(phi) * self.sigma_x + np.sin(phi) * self.sigma_z)
        gate = sp.linalg.expm(-1j * (theta * 2) * exponent)
        return gate


    def full_qubit_xy_rotation(self, theta, phi):
        """
        Generate the Qubit XY Rotation Gate in the Qubit-Qumode Basis
        Args:
            theta: rotation angle
            phi: rotation axis
        Returns:
            full_gate: ndarray of shape (2 * qumode_dim, 2 * qumode_dim)
        """
        full_gate = np.kron(self.qubit_xy_rotation(theta, phi), self.I_o)
        return full_gate
    

    def morse_hamiltonian(self, mp):
        """
        Generate the Morse Hamiltonian in the Qumode Basis
        Args:
            mp: MorsePotential object
        Returns:
            hamiltonian: ndarray of shape (qumode_dim, qumode_dim)
        """
        kinetic_part = (self.p_op @ self.p_op) 
        exponent_term = -1 * mp.b * (self.x_op - mp.x0 * self.I_o)
        position_part = mp.de * ((self.I_o - sp.linalg.expm(exponent_term)) @ (self.I_o - sp.linalg.expm(exponent_term)))
        hamiltonian = kinetic_part + position_part

        return hamiltonian
    

    def full_morse_hamiltonian(self, mp):
        """
        Generate the Morse Hamiltonian in the Qubit-Qumode Basis
        Args:
            mp: MorsePotential object
        Returns:
            full_hamiltonian: ndarray of shape (2 * qumode_dim, 2 * qumode_dim)
        """
        full_hamiltonian = np.kron(self.I_q, self.morse_hamiltonian(mp))
        return full_hamiltonian
    

    def morse_time_evolution(self, mp):
        """
        Generate the Time Evolution of the Morse Hamiltonian in the Qumode Basis
        Args:
            mp: MorsePotential object
        Returns:
            time_evolution: ndarray of shape (qumode_dim, qumode_dim)
        """
        time_evolution = sp.linalg.expm(-1j * self.morse_hamiltonian(mp) * self.time)
        return time_evolution
    

    def full_morse_time_evolution(self, mp):
        """
        Generate the Time Evolution of the Morse Hamiltonian in the Qubit-Qumode Basis
        Args:
            mp: MorsePotential object
        Returns:
            full_time_evolution: ndarray of shape (2 * qumode_dim, 2 * qumode_dim)
        """
        full_time_evolution = np.kron(self.I_q, self.morse_time_evolution(mp))
        return full_time_evolution
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------