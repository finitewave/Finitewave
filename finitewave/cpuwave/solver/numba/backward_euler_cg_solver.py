import numpy as np
import warnings

from ..solver import Solver
from .numba_linalg import cg_numba, matvec_numba


class BackwardEulerCGSolver(Solver):
    def __init__(self):
        self.maxiter = 100
        self.rtol = 0.
        self.atol = 1.e-8
        self.num_iterations = []
        self.b = None
        self.u = None
        self.myo_indexes = None

    def initialize(self, simulation):
        self.simulation = simulation
        self.num_iterations = []
        self.b = np.zeros_like(simulation.cardiac_model.u)
        self.u = simulation.cardiac_model.u
        self.myo_indexes = simulation.cardiac_model.myo_indexes
        self.assemble_system()

    def assemble_system(self):
        """Assembles the system matrix for the Backward Euler method."""
        stiff, mass = self.simulation.diffusion_model.weights
        dt = self.simulation.dt
        self.a_matrix = mass + dt * stiff
        self.mass_matrix = mass

    def run(self):
        self.u = (self.simulation.cardiac_model.u +
                  self.simulation.cardiac_model.rhs)
        self.b = matvec_numba(self.mass_matrix.indptr,
                              self.mass_matrix.indices,
                              self.mass_matrix.data, self.u, self.b,
                              self.myo_indexes)

        self.u, success = cg_numba(self.a_matrix.indptr, self.a_matrix.indices,
                                   self.a_matrix.data, self.b, self.u,
                                   self.myo_indexes, atol=self.atol,
                                   maxiter=self.maxiter)
        if success < 0:
            warnings.warn("Diffusion kernel solution accuracy is not reached")

        self.num_iterations.append(success)
        self.simulation.cardiac_model.u = self.u
        return self.u
