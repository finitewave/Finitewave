from scipy import sparse
import mlx.core as mx
import numpy as np
from finitewave.mlxwave.numerics.linalg.mlx_solvers import forward_euler_mlx
from .solver import Solver


class ForwardEulerSolverMlx(Solver):
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

    def initialize(self, simulation):
        """Initializes the Forward Euler solver with the given simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object containing the cardiac and diffusion models.
        """
        self.simulation = simulation
        self.u = simulation.cardiac_model.u
        self.u_old = mx.zeros(self.u.shape, dtype=mx.float32)
        self.rhs = simulation.cardiac_model.rhs
        self.myo_indexes = simulation.cardiac_model.myo_indexes
        self.assemble_system()

    def assemble_system(self):
        """Assembles the system matrix for the Forward Euler method.
        
        A_lhs = dt * M^{-1} * K

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
        self.a_lhs_matrix = dt * mass_inv * stiff
        self.a_lhs_matrix = self.build_ellpack(self.a_lhs_matrix)

        # print(self.a_lhs_matrix[0], self.a_lhs_matrix[1])

    def run(self):
        """Performs a single time step using the Forward Euler method.

        For each time step:
            1. Update the solution vector and right-hand side from the cardiac model.
            2. u_new = u - A_lhs @ u + dt * rhs (explicit diffusion step).
            3. Update the cardiac model solution with the new values.
        """        
        self.u = self.simulation.cardiac_model.u
        self.rhs = self.simulation.cardiac_model.rhs
        self.myo_indexes = self.simulation.cardiac_model.myo_indexes

        self.u_old = forward_euler_mlx(*self.a_lhs_matrix, self.u, self.u_old, self.rhs,
                                       self.myo_indexes, self.simulation.dt)
        mx.eval(self.u_old)
        mx.synchronize()
        self.u_old, self.u = self.u, self.u_old
        self.simulation.cardiac_model.u = self.u
        self.num_iterations.append(1)

    def build_ellpack(self, csr_matrix):
        """Converts a sparse matrix in CSR format to ELLPACK format.

        Parameters
        ----------
        csr_matrix : scipy.sparse.csr_matrix
            The input sparse matrix in CSR format.

        Returns
        -------
        indices : mx.ndarray
            The column indices of the non-zero elements in ELLPACK format.
        data : mx.ndarray
            The non-zero values of the matrix in ELLPACK format.
        """
        row_lengths = np.diff(csr_matrix.indptr)
        K = np.max(row_lengths)
        M = csr_matrix.shape[0]

        ellpack_indices = np.repeat(np.arange(M), K).reshape(M, K)
        ellpack_data = np.zeros((M, K), dtype=np.float32)

        inds = np.repeat([np.arange(K)], M, axis=0)
        mask = inds < row_lengths[:, None]
        ellpack_indices[mask] = csr_matrix.indices
        ellpack_data[mask] = csr_matrix.data.astype(np.float32)

        ellpack_indices = mx.array(ellpack_indices, dtype=mx.int32)
        ellpack_data = mx.array(ellpack_data, dtype=mx.float32)

        print(ellpack_data[:5])

        return ellpack_indices, ellpack_data
