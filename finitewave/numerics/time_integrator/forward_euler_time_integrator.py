import numpy as np
from scipy import sparse
from finitewave.core.numerics.time_integrator import TimeIntegrator



class ForwardEulerTimeIntegrator(TimeIntegrator):
    """Implements the Forward Euler time integration method for cardiac
    simulations.

    Attributes
    ----------
    full_lumping : bool
        If True, uses full lumping of the mass matrix in the diffusion model.
    a_matrix : scipy.sparse.csr_matrix
        The system matrix for the Forward Euler method.
    u_new : np.ndarray
        The solution vector at the new time step.
    u : np.ndarray
        The solution vector at the current time step.
    rhs : np.ndarray
        The right-hand side vector from the cardiac model.
    myo_mask : np.ndarray
        A boolean mask indicating the myocyte cells in the simulation.
    num_iterations : list
        List to track the number of iterations per time step.
    """
    def __init__(self, ionic_lumping=False):
        self.a_matrix = None
        self.u_new = None
        self.u = None
        self.rhs = None
        self.myo_mask = None
        self.num_iterations = []
        self.solver = None
        self.ionic_lumping = ionic_lumping

    def initialize(self, simulation):
        """Initializes the Forward Euler solver with the given simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object containing the cardiac and diffusion models.
        """
        self.simulation = simulation
        self.linalg = simulation.backend.linalg

        self.num_iterations = []
        self.u = simulation.cardiac_model.u
        self.u_old = simulation.backend.copy(self.u)
        self.rhs = simulation.cardiac_model.rhs
        self.assemble_system()

    def assemble_system(self):
        """Assembles the system matrix for the Forward Euler method.
        
        A_lhs = I - dt * M_lumped^{-1} * K
        A_ion = dt * M_lumped^{-1} * M_ion
        If ionic_lumping is True, M_ion = M_lumped, otherwise M_ion = M (full mass matrix).

        Parameters
        ----------
        stiffness_matrix : scipy.sparse.csr_matrix
            The stiffness matrix from the diffusion model.
        mass_matrix : scipy.sparse.csr_matrix
            The mass matrix from the diffusion model.
        dt : float
            Time step for the simulation.
        """
        dt = self.simulation.dt
        myo_indexes = self.simulation.cardiac_tissue.myo_indexes
        stiff, mass = self.simulation.spatial_discretization.weights

        a_rhs_matrix = self.assemble_rhs_matrix(stiff, mass, dt)
        self.a_rhs_matrix = self.simulation.backend.wrap_sparse(
            a_rhs_matrix, myo_indexes, row_reduced=False)

        a_ion_matrix = self.assemble_ion_matrix(mass, dt)
        self.a_ion_matrix = self.simulation.backend.wrap_sparse(
            a_ion_matrix, myo_indexes, row_reduced=False)

        if self.solver is None:
            self.solver = self.linalg.select_explicit_solver(self.u, myo_indexes)

        myo_mask = np.zeros_like(self.u, dtype=bool)
        myo_mask[myo_indexes] = True
        self.myo_mask = self.simulation.backend.wrap_mask(myo_mask)

    def assemble_rhs_matrix(self, stiff, mass, dt):
        """Assembles the right-hand side matrix for the Forward Euler method.

        Parameters
        ----------
        stiff : scipy.sparse.csr_matrix
            The stiffness matrix from the diffusion model.
        mass : scipy.sparse.csr_matrix
            The mass matrix from the diffusion model.
        dt : float
            Time step for the simulation.
        """
        mass_lumped_inv = sparse.diags(1 / mass.sum(axis=1).A1)
        a_rhs_matrix = sparse.eye(stiff.shape[0]) - dt * mass_lumped_inv * stiff
        return a_rhs_matrix

    def assemble_ion_matrix(self, mass, dt):
        """Assembles the ionic current matrix for the Forward Euler method.

        Parameters
        ----------
        mass : scipy.sparse.csr_matrix
            The mass matrix from the diffusion model.
        dt : float
            Time step for the simulation.

        Returns
        -------
        scipy.sparse.csr_matrix
            The ionic current matrix for the Forward Euler method.
            If ionic_lumping is True, returns dt * I,
            otherwise returns dt * M_lumped^{-1} * M.
        """
        if self.ionic_lumping:
            return dt * sparse.eye(mass.shape[0])
        
        mass_lumped_inv = sparse.diags(1 / mass.sum(axis=1).A1)
        return dt * mass_lumped_inv @ mass

    def run(self):
        """Performs a single time step using the Forward Euler method.

        For each time step:
            1. Update the solution vector and right-hand side from the cardiac model.
            2. u_new = (I - dt * M^{-1} * K) @ u + dt * rhs (explicit diffusion step).
            3. Update the cardiac model solution with the new values.
        """        
        self.u = self.simulation.cardiac_model.u
        self.rhs = self.simulation.cardiac_model.rhs

        self.u_old, self.u = self.u, self.u_old

        self.u = self.solver(
            self.a_rhs_matrix,
            self.u_old,
            self.a_ion_matrix,
            self.rhs,
            self.myo_mask, 
            self.u
        )

        self.simulation.cardiac_model.u = self.u
        self.num_iterations.append(1)
