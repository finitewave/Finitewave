from scipy import sparse
import numpy as np
import mlx.core as mx
from .linalg.mlx_solvers import forward_euler_mlx
from .solver import Solver


class ForwardEulerSolver(Solver):
    """Implements the Forward Euler time integration method for cardiac
    simulations.

    Attributes
    ----------
    a_matrix : scipy.sparse.csr_matrix
        The system matrix for the Forward Euler method.
    u_old : np.ndarray
        The solution vector at the previous time step.
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
        self.u_old = None
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
        self.a_lhs_matrix = self.build_ellpack(self.a_lhs_matrix)

    def run(self):
        """Performs a single time step using the Forward Euler method.

        For each time step:
            1. Update the solution vector and right-hand side from the cardiac model.
            2. u = A_lhs @ u_old + dt * rhs (explicit diffusion step).
            3. Update the cardiac model solution with the new values.
        """        
        self.u = self.simulation.cardiac_model.u
        self.rhs = self.simulation.cardiac_model.rhs
        self.myo_indexes = self.simulation.cardiac_model.myo_indexes
        
        self.u_old, self.u = self.u, self.u_old

        self.u = forward_euler_mlx(*self.a_lhs_matrix, self.u, self.u_old, self.rhs,
                                   self.myo_indexes, self.simulation.dt)
        
        if self.simulation.step % 10 == 0:
            mx.eval(self.u)
            mx.synchronize()

        self.simulation.cardiac_model.u = self.u
        self.num_iterations.append(1)
