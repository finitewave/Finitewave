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
        self.method = None

    def initialize(self, simulation):
        """Initializes the Crank-Nicolson CG solver with the given simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object containing the cardiac and diffusion models.
        """
        self.simulation = simulation
        self.num_iterations = []
        self.b = np.zeros_like(simulation.cardiac_model.u)
        self.u = simulation.cardiac_model.u
        self.u_new = self.u.copy()
        self.myo_indexes = simulation.cardiac_model.myo_indexes
        self.rhs = simulation.cardiac_model.rhs
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

        self.u_old, self.u = self.u, self.u_old  # Swap references for in-place updates
        # Explicit step for the reaction term (rhs of ionic model)
        self.u = self.method.cg(self.simulation.dt, self.rhs, self.u,
                                self.myo_indexes)
        # Implicit step for the diffusion term
        self.b = matvec_numba(self.a_rhs_matrix.indptr,
                              self.a_rhs_matrix.indices,
                              self.a_rhs_matrix.data, self.u, self.b,
                              self.myo_indexes)
        self.u, success = cg_numba(self.a_lhs_matrix, self.b, self.u,
                                   self.myo_indexes, atol=self.atol,
                                   maxiter=self.maxiter)
        if success < 0:
            warnings.warn("Diffusion kernel solution accuracy is not reached")

        self.num_iterations.append(success)
        self.simulation.cardiac_model.u = self.u
        return self.u

