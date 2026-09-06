import numpy as np
from scipy import sparse
from finitewave.core.numerics.time_integration import TimeIntegration



class ForwardEulerTimeIntegration(TimeIntegration):
    """Implements the Forward Euler time integration method for cardiac simulations.

    Attributes
    ----------
    a_rhs_matrix : sparse matrix
        The right-hand side matrix for the Forward Euler method.
    a_reaction_matrix : sparse matrix
        The reaction matrix for the Forward Euler method.
    u_old : np.ndarray
        The solution vector at the previous time step.
    u : np.ndarray
        The solution vector at the current time step.
    reaction_term : np.ndarray
        The reaction term computed by the cardiac model.
    myo_mask : np.ndarray
        A boolean mask indicating the myocyte cells in the simulation.
    num_iterations : list
        List to track the number of iterations per time step.
    solver : callable
        The solver function for the explicit diffusion step.
    linalg : linalg backend
        The linear algebra backend used for matrix operations.
    reaction_lumping : bool
        If True, uses lumping for the reaction term; otherwise, uses the full
        mass matrix.
    """
    def __init__(self, reaction_lumping=False):
        self.a_rhs_matrix = None
        self.a_reaction_matrix = None
        self.u_old = None
        self.u = None
        self.reaction_term = None
        self.myo_mask = None
        self.num_iterations = []
        self.solver = None
        self.linalg = None
        self.reaction_lumping = reaction_lumping

    def initialize(self, simulation):
        """Initializes Forward Euler time integration with the given simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation containing the cardiac model, tissue, backend, and
            spatial discretization.
        """
        self.simulation = simulation
        self.linalg = simulation.backend.linalg

        self.num_iterations = []
        self.u = simulation.cardiac_model.u
        self.u_old = simulation.backend.copy(self.u)
        self.reaction_term = simulation.cardiac_model.rhs
        self.assemble_system()

    def assemble_system(self):
        """Assembles the system matrices and selects the Forward Euler solver.

        A_rhs = I - dt * M_lumped^{-1} * K
        A_reaction = dt * I                              if reaction_lumping
        A_reaction = dt * M_lumped^{-1} * M              otherwise
        """
        dt = self.simulation.dt
        myo_indexes = self.simulation.cardiac_tissue.myo_indexes
        stiff, mass = self.simulation.spatial_discretization.weights

        a_rhs_matrix = self.assemble_rhs_matrix(stiff, mass, dt)
        self.a_rhs_matrix = self.simulation.backend.wrap_sparse(
            a_rhs_matrix, myo_indexes, row_reduced=False)

        a_reaction_matrix = self.assemble_reaction_matrix(mass, dt)
        self.a_reaction_matrix = self.simulation.backend.wrap_sparse(
            a_reaction_matrix, myo_indexes, row_reduced=False)

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

        Returns
        -------
        scipy.sparse.csr_matrix
            The right-hand side matrix for the Forward Euler method.
        """
        mass_lumped_inv = sparse.diags(1 / mass.sum(axis=1).A1)
        a_rhs_matrix = sparse.eye(stiff.shape[0]) - dt * mass_lumped_inv * stiff
        return a_rhs_matrix

    def assemble_reaction_matrix(self, mass, dt):
        """Assembles the reaction matrix for the Forward Euler method.

        Parameters
        ----------
        mass : scipy.sparse.csr_matrix
            The mass matrix from the diffusion model.
        dt : float
            Time step for the simulation.

        Returns
        -------
        scipy.sparse.csr_matrix
            The reaction matrix for the Forward Euler method.
            If reaction_lumping is True, returns dt * I,
            otherwise returns dt * M_lumped^{-1} * M.
        """
        if self.reaction_lumping:
            return dt * sparse.eye(mass.shape[0])
        
        mass_lumped_inv = sparse.diags(1 / mass.sum(axis=1).A1)
        return dt * mass_lumped_inv @ mass

    def run(self):
        """Performs a single time integration step using the Forward Euler method.

        For each time step:
            1. Update the solution vector and reaction term from the cardiac model.
            2. Swap the old and current solution vectors.
            3. Compute u = A_rhs @ u_old + A_reaction @ reaction_term.
            4. Update the cardiac model solution with the new values.
        """
        self.u = self.simulation.cardiac_model.u
        self.reaction_term = self.simulation.cardiac_model.rhs

        self.u_old, self.u = self.u, self.u_old

        self.u = self.solver(
            self.a_rhs_matrix,
            self.u_old,
            self.a_reaction_matrix,
            self.reaction_term,
            self.myo_mask, 
            self.u
        )

        self.simulation.cardiac_model.u = self.u
        self.num_iterations.append(1)
