import numpy as np
import warnings

from ..solver import Solver
from .numba_linalg import cg_numba, matvec_numba


class CrankNicolsonCGSolver(Solver):
    def __init__(self):
        self.maxiter = 100
        self.atol = 1e-8
        self.num_iterations = []
        self.b0 = None
        self.b1 = None
        self.u = None
        self.a_lhs_matrix = None
        self.a_rhs_matrix = None
        self.mass_matrix = None

    def initialize(self, simulation):
        self.simulation = simulation
        self.num_iterations = []
        self.b0 = np.zeros_like(simulation.cardiac_model.u)
        self.b1 = np.zeros_like(simulation.cardiac_model.u)
        self.u = simulation.cardiac_model.u
        self.myo_indexes = simulation.cardiac_model.myo_indexes
        self.rhs = simulation.cardiac_model.rhs
        self.assemble_system()

    def assemble_system(self):
        stiff, mass = self.simulation.diffusion_model.weights
        dt = self.simulation.dt
        self.a_lhs_matrix = mass + 0.5 * dt * stiff
        self.a_rhs_matrix = mass - 0.5 * dt * stiff
        self.mass_matrix = mass

    def run(self):
        self.u = self.simulation.cardiac_model.u
        self.rhs = self.simulation.cardiac_model.rhs
        self.myo_indexes = self.simulation.cardiac_model.myo_indexes

        self.b0 = matvec_numba(self.a_rhs_matrix.indptr,
                               self.a_rhs_matrix.indices,
                               self.a_rhs_matrix.data, self.u, self.b0,
                               self.myo_indexes)
        self.b1 = matvec_numba(self.mass_matrix.indptr,
                               self.mass_matrix.indices,
                               self.mass_matrix.data, self.rhs, self.b1,
                               self.myo_indexes)
        b = self.b0 + self.b1
        self.u, success = cg_numba(self.a_lhs_matrix.indptr,
                                   self.a_lhs_matrix.indices,
                                   self.a_lhs_matrix.data, b, self.u,
                                   self.myo_indexes, atol=self.atol,
                                   maxiter=self.maxiter)
        if success < 0:
            warnings.warn("Diffusion kernel solution accuracy is not reached")

        self.num_iterations.append(success)
        self.simulation.cardiac_model.u = self.u
        return self.u
