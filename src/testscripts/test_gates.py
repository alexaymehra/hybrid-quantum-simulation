"""
Filename: test_gates.py
Author: Alexay Mehra
Date: 2025-10-16
Description: Tests for gate implementations
"""

import numpy as np
import scipy as sp
import math
from qutip import destroy

from src.optimization import infidelity
import optimization, simulation, utils, visualization

def test_controlled_rotation():
    omega = 0.0
    chi = 2.0
    backend = utils.Gates(dim = 20, time = 1.0)
    created_by_gates = backend.c_r_ham(omega = omega, chi = chi)

    a = destroy(backend.dim) # annihilation operator
    adag = a.dag()

    CR = (chi * sz)*(a.dag()*a)
    expected = 