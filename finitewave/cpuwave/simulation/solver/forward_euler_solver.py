import numpy as np
from finitewave.cpuwave.numerics.linalg.solvers import forward_euler
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

    def initialize(self, simulation):
        """Initializes the Forward Euler solver with the given simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object containing the cardiac and diffusion models.
        """
        self.simulation = simulation
        self.u_new = simulation.cardiac_model.u.copy()
        self.u = simulation.cardiac_model.u
        self.rhs = simulation.cardiac_model.rhs
        self.myo_indexes = simulation.cardiac_model.myo_indexes
        self.assemble_system()

    def assemble_system(self):
        """Assembles the system matrix for the Forward Euler method.
        Takes the stiffness and mass matrices from the diffusion model
        and computes the inverse of the lumped mass matrix diagonal.

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
        # stiff, mass, nonzero_rows = self.remove_zeros(stiff, mass)
        mass_lumped = mass.sum(axis=1).A.ravel()
        self.mass_inv = 1 / mass_lumped
        self.stiff = stiff
        # self.myo_indexes = self.myo_indexes[nonzero_rows]

    def run(self):
        """Performs a single time step using the Forward Euler method.

        For each time step:
        1. Update the solution vector and right-hand side from the cardiac model.
        2. u_new = u - dt * M^{-1} * K * u + dt * rhs (explicit diffusion step).
        3. Update the cardiac model solution with the new values.
        """        
        self.u = self.simulation.cardiac_model.u
        self.rhs = self.simulation.cardiac_model.rhs
        self.myo_indexes = self.simulation.cardiac_model.myo_indexes

        forward_euler(self.stiff, self.u, self.rhs, self.mass_inv,
                      self.u_new, self.myo_indexes, self.simulation.dt)

        self.u, self.u_new = self.u_new, self.u
        self.simulation.cardiac_model.u = self.u
        self.num_iterations.append(1)

    def remove_zeros(self, stiff, mass):
        """Removes zero rows from the stiffness and mass matrices to optimize
        the system for the Forward Euler method.

        Parameters
        ----------
        stiff : scipy.sparse.csr_matrix
            The stiffness matrix to be modified.
        mass : scipy.sparse.csr_matrix
            The mass matrix to be modified.

        Returns
        -------
        scipy.sparse.csr_matrix
            The modified stiffness matrix with zero rows removed.
        scipy.sparse.csr_matrix
            The modified mass matrix with zero rows removed.
        np.ndarray
            The indices of the non-zero rows in the original stiffness matrix.
        """
        nonzero_rows = np.where(stiff.getnnz(axis=1) > 0)[0]
        stiff = stiff[nonzero_rows][:, nonzero_rows]
        mass = mass[nonzero_rows][:, nonzero_rows]
        return stiff, mass, nonzero_rows
