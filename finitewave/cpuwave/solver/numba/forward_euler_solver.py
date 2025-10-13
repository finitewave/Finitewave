from .numba_linalg import matvec_and_add_numba
from ..solver import Solver


class ForwardEulerSolver(Solver):
    def __init__(self):
        self.a_matrix = None
        self.u_new = None
        self.u = None
        self.rhs = None
        self.myo_indexes = None

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
        dt = self.simulation.dt
        self.a_matrix = mass + dt * stiff

    def run(self):
        self.u = self.simulation.cardiac_model.u
        self.rhs = self.simulation.cardiac_model.rhs
        self.myo_indexes = self.simulation.cardiac_model.myo_indexes

        matvec_and_add_numba(self.a_matrix.indptr, self.a_matrix.indices,
                             self.a_matrix.data, self.u, self.rhs, self.u_new,
                             self.myo_indexes)

        self.u, self.u_new = self.u_new, self.u
        self.simulation.cardiac_model.u = self.u
