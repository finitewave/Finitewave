import numpy as np
from scipy import sparse
from .solver import Solver


class ForwardEulerSolver(Solver):
    """Implements the Forward Euler time integration method for cardiac
    simulations.

    Attributes
    ----------
    a_matrix : scipy.sparse.csr_matrix
        The system matrix for the Forward Euler method.
    u_new : np.ndarray
        The solution vector at the new time step.
    u : np.ndarray
        The solution vector at the current time step.
    rhs : np.ndarray
        The right-hand side vector from the cardiac model.
    myo_indexes : np.ndarray
        Indexes of myocardial nodes in the simulation.
    num_iterations : list
        List to track the number of iterations per time step.
    """
    def __init__(self):
        self.a_matrix = None
        self.u_new = None
        self.u = None
        self.rhs = None
        self.myo_indexes = None
        self.num_iterations = []
        self.linalg_method = None

    def initialize(self, simulation):
        """Initializes the Forward Euler solver with the given simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object containing the cardiac and diffusion models.
        """
        self.simulation = simulation
        self.u_old = simulation.backend.copy(simulation.cardiac_model.u)
        self.u = simulation.cardiac_model.u
        self.rhs = simulation.cardiac_model.rhs
        self.myo_indexes = simulation.cardiac_model.myo_indexes
        self.assemble_system()
        self.select_method(simulation.backend)

    def select_method(self, backend):
        if backend.name == "numba":
            from .numba_linalg.numba_methods import NumbaEuler
            self.linalg_method = NumbaEuler()
            return

        if backend.name == "mlx":
            from .mlx_linalg.mlx_methods import MlxEuler
            self.linalg_method = MlxEuler()
            return
        
        if backend.name == "jax":
            from .jax_linalg.jax_methods import JaxEuler
            self.linalg_method = JaxEuler()
            return

    def assemble_system(self):
        """Assembles the system matrix for the Forward Euler method.
        
        A_lhs = I - dt * M^{-1} * K

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
        mass_lumped = mass.sum(axis=1).A.ravel()
        mass_inv = sparse.diags(1 / mass_lumped)
        self.a_lhs_matrix = sparse.eye(stiff.shape[0]) - dt * mass_inv * stiff
        
        if self.simulation.backend.sparse_support:
            self.a_lhs_matrix = self.crs_to_numpy(self.a_lhs_matrix)
        else:
            self.a_lhs_matrix = self.csr_to_ellpack(self.a_lhs_matrix)

    def run(self):
        """Performs a single time step using the Forward Euler method.

        For each time step:
            1. Update the solution vector and right-hand side from the cardiac model.
            2. u_new = u - dt * M^{-1} * K @ u + dt * rhs (explicit diffusion step).
            3. Update the cardiac model solution with the new values.
        """        
        self.u = self.simulation.cardiac_model.u
        self.rhs = self.simulation.cardiac_model.rhs
        self.myo_indexes = self.simulation.cardiac_model.myo_indexes

        self.u_old, self.u = self.u, self.u_old
        
        self.u = self.linalg_method.solve(*self.a_lhs_matrix, self.u_old, self.rhs,
                                          self.simulation.dt, self.myo_indexes, self.u)
        self.u = self.linalg_method.evaluate(self.u, self.simulation.step)
        self.simulation.cardiac_model.u = self.u
        self.num_iterations.append(1)
