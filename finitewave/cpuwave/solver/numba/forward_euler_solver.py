from .numba_linalg import forward_euler, matvec_numba, ax_p_y_numba
from ..solver import Solver
import scipy.sparse as sparse
import numpy as np


class ForwardEulerSolver(Solver):
    def __init__(self):
        self.a_matrix = None
        self.u_new = None
        self.u = None
        self.rhs = None
        self.myo_indexes = None
        self.num_iterations = []

    def initialize(self, simulation):
        self.simulation = simulation
        self.u_new = simulation.cardiac_model.u.copy()
        self.u = simulation.cardiac_model.u
        self.rhs = simulation.cardiac_model.rhs
        self.myo_indexes = simulation.cardiac_model.myo_indexes
        self.assemble_system()

    def assemble_system(self):
        """Assembles the system matrix for the Forward Euler method.
        A = M + dt * K.

        TODO: Add support for mass lumping.

        Parameters
        ----------
        stiffness_matrix : scipy.sparse.csr_matrix
            The stiffness matrix from the diffusion model.
        mass_matrix : scipy.sparse.csr_matrix
            The mass matrix from the diffusion model.
        dt : float
            Time step for the simulation.
        """
        stiff, mass = self.simulation.diffusion_model.weights
        mass_lumped = mass.sum(axis=1).A.ravel()
        self.mass_inv = 1 / mass_lumped
        self.stiff = stiff

    def run(self):
        self.u = self.simulation.cardiac_model.u
        self.rhs = self.simulation.cardiac_model.rhs
        self.myo_indexes = self.simulation.cardiac_model.myo_indexes

        forward_euler(self.stiff.indptr, self.stiff.indices,
                      self.stiff.data, self.u, self.rhs, self.mass_inv,
                      self.u_new, self.myo_indexes, self.simulation.dt)

        self.u, self.u_new = self.u_new, self.u
        self.simulation.cardiac_model.u = self.u
        self.num_iterations.append(1)
