import numpy as np
import warnings

from ..solver import Solver
from .numba_linalg import cg_numba, matvec_numba


class CrankNicolsonCGSolver(Solver):
    def __init__(self):
        self.maxiter = 100
        self.atol = 1e-8
        self.num_iterations = []

    def initialize(self, u):
        self.num_iterations = []
        self.b0 = np.zeros_like(u)
        self.b1 = np.zeros_like(u)

    def assemble_system(self, stiffness_matrix, mass_matrix, dt):
        a_lhs_matrix = mass_matrix + 0.5 * dt * stiffness_matrix
        a_rhs_matrix = mass_matrix - 0.5 * dt * stiffness_matrix
        mass_matrix = mass_matrix
        return [a_lhs_matrix, a_rhs_matrix, mass_matrix]

    def solve(self, u, rhs, indexes, matrices):
        a_lhs_matrix = matrices[0]
        a_rhs_matrix = matrices[1]
        mass_matrix = matrices[2]
        self.b0 = matvec_numba(a_rhs_matrix.indptr, a_rhs_matrix.indices,
                               a_rhs_matrix.data, u, self.b0, indexes)
        self.b1 = matvec_numba(mass_matrix.indptr, mass_matrix.indices,
                               mass_matrix.data, rhs, self.b1, indexes)
        b = self.b0 + self.b1
        u, success = cg_numba(a_lhs_matrix.indptr, a_lhs_matrix.indices,
                              a_lhs_matrix.data, b, u, indexes, atol=self.atol,
                              maxiter=self.maxiter)
        if success < 0:
            warnings.warn("Diffusion kernel solution accuracy is not reached")
        self.num_iterations.append(success)
        return u
