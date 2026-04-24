import numpy as np
import warnings

from .solver import Solver


class CrankNicolsonSolver(Solver):
    """Implements the Crank-Nicolson semi-implicit time integration method
    with Conjugate Gradient solver for implicit diffusion step.

    Attributes
    ----------
    maxiter : int
        Maximum number of iterations for the CG solver.
    atol : float
        Absolute tolerance for the CG solver.
    num_iterations : list
        List to track the number of iterations per time step.
    b : np.ndarray
        The right-hand side vector for the linear system.
    u : np.ndarray
        The solution vector at the current time step.
    a_lhs_matrix : scipy.sparse.csr_matrix
        The left-hand side system matrix for the Crank-Nicolson method.
    a_rhs_matrix : scipy.sparse.csr_matrix
        The right-hand side system matrix for the Crank-Nicolson method.
    mass_matrix : scipy.sparse.csr_matrix
        The mass matrix from the diffusion model.
    """
    def __init__(self):
        self.maxiter = 100
        self.atol = 1e-8
        self.num_iterations = []
        self.b = None
        self.u = None
        self.a_lhs_matrix = None
        self.a_rhs_matrix = None
        self.mass_matrix = None
        self.linalg_method = None

    def initialize(self, simulation):
        """Initializes the Crank-Nicolson CG solver with the given simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object containing the cardiac and diffusion models.
        """
        self.simulation = simulation
        self.num_iterations = []
        self.u = simulation.cardiac_model.u
        self.b = simulation.backend.wrap(0. * self.u)
        self.u_old = simulation.backend.copy(self.u)
        self.rhs = simulation.cardiac_model.rhs
        self.myo_indexes = simulation.cardiac_model.myo_indexes
        self.assemble_system()
        if self.linalg_method is None:
            self.select_method(simulation.backend)

    def select_method(self, backend):
        if backend.name == "numba":
            from .numba_linalg.numba_methods import NumbaCG
            self.linalg_method = NumbaCG()
            return

        if backend.name == "mlx":
            from .mlx_linalg.mlx_methods import MlxCG
            self.linalg_method = MlxCG()
            return
        
        if backend.name == "jax":
            from .jax_linalg.jax_methods import JaxCG
            self.linalg_method = JaxCG()
            return

    def assemble_system(self):
        """Assembles the system matrices for the Crank-Nicolson method.
        
        A_lhs = M + 0.5 * dt * K
        A_rhs = M - 0.5 * dt * K
        """
        stiff, mass = self.simulation.diffusion_model.weights
        dt = self.simulation.dt
        self.a_lhs_matrix = mass + 0.5 * dt * stiff
        self.a_rhs_matrix = mass - 0.5 * dt * stiff

        if self.simulation.backend.sparse_support:
            self.a_lhs_matrix = self.crs_to_numpy(self.a_lhs_matrix)
            self.a_rhs_matrix = self.crs_to_numpy(self.a_rhs_matrix)
        else:
            self.a_lhs_matrix = self.csr_to_ellpack(self.a_lhs_matrix)
            self.a_rhs_matrix = self.csr_to_ellpack(self.a_rhs_matrix)

    def run(self):
        """Performs a single time step using the Crank-Nicolson method
        with Conjugate Gradient solver for the implicit diffusion step.

        For each time step:
            1. Update the solution vector and right-hand side from the cardiac model.
            2. u = u + dt * rhs (explicit reaction step).
            3. b = A_rhs @ u (formulate the right-hand side for diffusion).
            4. Solve A_lhs @ u_new = b using Conjugate Gradient method.
            5. Update the cardiac model solution with the new values.
        """
        self.u = self.simulation.cardiac_model.u
        self.rhs = self.simulation.cardiac_model.rhs
        self.myo_indexes = self.simulation.cardiac_model.myo_indexes
        # Swap references for in-place updates
        self.u_old, self.u = self.u, self.u_old
        # Explicit step for the reaction term (rhs of ionic model)
        self.u = self.linalg_method.axpy(self.simulation.dt, self.rhs, self.u_old,
                                         self.myo_indexes, self.u)
        # Implicit step for the diffusion term
        self.b = self.linalg_method.matvec(*self.a_rhs_matrix, self.u,
                                           self.myo_indexes, self.b)
        self.u, n_iter = self.linalg_method.solve(*self.a_lhs_matrix, self.b, 
                                                  self.u, self.myo_indexes,
                                                  atol=self.atol, maxiter=self.maxiter)
        if n_iter < 0:
            warnings.warn("Diffusion kernel solution accuracy is not reached")

        self.num_iterations.append(n_iter)
        self.simulation.cardiac_model.u = self.u
        return self.u

