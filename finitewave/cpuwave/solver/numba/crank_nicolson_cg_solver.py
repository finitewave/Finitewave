import numpy as np
import warnings

from ..solver import Solver
from .numba_linalg import cg_numba, matvec_numba, ax_p_y_numba


class CrankNicolsonCGSolver(Solver):
    def __init__(self):
        self.maxiter = 100
        self.atol = 1e-8
        self.num_iterations = []
        self.b = None
        self.u = None
        self.a_lhs_matrix = None
        self.a_rhs_matrix = None
        self.mass_matrix = None

    def initialize(self, simulation):
        self.simulation = simulation
        self.num_iterations = []
        self.b = np.zeros_like(simulation.cardiac_model.u)
        self.u = simulation.cardiac_model.u
        self.myo_indexes = simulation.cardiac_model.myo_indexes
        self.rhs = simulation.cardiac_model.rhs
        self.assemble_system()

    def assemble_system(self):
        stiff, mass = self.simulation.diffusion_model.weights
        dt = self.simulation.dt
        self.a_lhs_matrix = mass + 0.5 * dt * stiff
        self.a_rhs_matrix = mass - 0.5 * dt * stiff

    def run(self):
        self.u = self.simulation.cardiac_model.u
        self.rhs = self.simulation.cardiac_model.rhs
        self.myo_indexes = self.simulation.cardiac_model.myo_indexes
        # Explicit step for the reaction term (rhs of ionic model)
        self.u = ax_p_y_numba(self.simulation.dt, self.rhs, self.u,
                              self.myo_indexes)
        # Implicit step for the diffusion term
        self.b = matvec_numba(self.a_rhs_matrix.indptr,
                              self.a_rhs_matrix.indices,
                              self.a_rhs_matrix.data, self.u, self.b,
                              self.myo_indexes)
        self.u, success = cg_numba(self.a_lhs_matrix.indptr,
                                   self.a_lhs_matrix.indices,
                                   self.a_lhs_matrix.data, self.b, self.u,
                                   self.myo_indexes, atol=self.atol,
                                   maxiter=self.maxiter)
        if success < 0:
            warnings.warn("Diffusion kernel solution accuracy is not reached")

        self.num_iterations.append(success)
        self.simulation.cardiac_model.u = self.u
        return self.u
