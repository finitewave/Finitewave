import numpy as np
import warnings

from finitewave.elementalwave.diffusion.solver.solver import Solver
from .numba_linalg import cg_numba, matvec_numba


class ImplicitEulerCGSolver(Solver):
    def __init__(self):
        self.maxiter = 100
        self.rtol = 0.
        self.atol = 1.e-8
        self.num_iterations = []

    def initialize(self, u):
        self.num_iterations = []
        self.b = np.zeros_like(u)

    def assemble_system(self, stiffness_matrix, mass_matrix, dt):
        a_matrix = mass_matrix + dt * stiffness_matrix
        return [a_matrix, mass_matrix]

    def solve(self, u, rhs, indexes, matrices):
        a_matrix = matrices[0]
        mass_matrix = matrices[1]
        temp = u + rhs
        self.b = matvec_numba(mass_matrix.indptr, mass_matrix.indices,
                              mass_matrix.data, temp, self.b, indexes)

        u, success = cg_numba(a_matrix.indptr, a_matrix.indices, a_matrix.data,
                              self.b, u, indexes, atol=self.atol,
                              maxiter=self.maxiter)
        if success < 0:
            warnings.warn("Diffusion kernel solution accuracy is not reached")
        self.num_iterations.append(success)
        return u
