import warnings
from scipy import sparse as sp
from finitewave.core.numerics.time_integration import TimeIntegration



class ImplicitTimeIntegration(TimeIntegration):
    """Implements implicit time integration for cardiac simulations.

    The diffusion system is solved with the Conjugate Gradient method. Mass
    lumping and reaction-term lumping can be configured independently.

    Attributes
    ----------
    maxiter : int
        Maximum number of iterations for the CG solver.
    atol : float
        Absolute tolerance for the CG solver.
    lumping_factor : float
        Interpolation factor between the consistent and lumped mass matrices.
        A value of 0 uses the consistent mass matrix, while 1 uses the fully
        lumped mass matrix.
    order : int
        Order of the implicit method. Use 1 for Backward Euler and 2 for
        Crank-Nicolson.
    reaction_lumping : bool
        If True, uses a lumped mass matrix for the reaction term; otherwise,
        uses the consistent mass matrix.
    num_iterations : list
        List to track the number of iterations per time step.
    u : np.ndarray
        The solution vector at the current time step.
    u_old : np.ndarray
        The solution vector at the previous time step.
    u_old_2 : np.ndarray
        The solution vector from two time steps ago.
    reaction_term : np.ndarray
        The reaction term computed by the cardiac model.
    a_lhs_matrix : scipy.sparse.csr_matrix
        The left-hand side system matrix for the implicit method.
        Shape should be (num_active_cells, num_active_cells) and
        indexing should be local to the active cells.
    a_rhs_matrix : scipy.sparse.csr_matrix
        The right-hand side system matrix for the implicit method.
    a_reaction_matrix : scipy.sparse.csr_matrix
        The matrix for the reaction contribution.
    solver : callable
        The iterative linear solver used for the implicit diffusion step.

    References
    ----------
    .. [1] Pathmanathan, Pras, et al. "Computational modelling of cardiac
           electrophysiology: explanation of the variability of results
           from different numerical solvers."
           International journal for numerical methods in biomedical
           engineering 28.8 (2012): 890-903.

    """
    def __init__(self, atol=1e-8, maxiter=100, lumping_factor=.0, order=1,
                 reaction_lumping=False):
        self.atol = atol
        self.maxiter = maxiter
        self.lumping_factor = lumping_factor
        self.order = order
        self.reaction_lumping = reaction_lumping

        self.u = None
        self.u_old = None
        self.u_old_2 = None

        self.a_lhs_matrix = None
        self.a_rhs_matrix = None
        self.a_reaction_matrix = None

        self.linalg = None
        self.solver = None

        self.num_iterations = []

    def initialize(self, simulation):
        """Initializes implicit time integration with the given simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation containing the cardiac model, tissue, backend, and
            spatial discretization.
        """
        self.simulation = simulation

        self.linalg = simulation.backend.linalg

        if self.solver is None:
            self.solver = self.linalg.cg

        if self.order not in [1, 2]:
            raise ValueError("ImplicitTimeIntegration order must be 1 or 2.")

        self.num_iterations = []
        self.u = simulation.cardiac_model.u
        self.u_old = simulation.backend.copy(self.u)
        self.u_old_2 = simulation.backend.copy(self.u)
        self.reaction_term = simulation.cardiac_model.rhs
        self.assemble_system()

    def assemble_system(self):
        """Assembles the matrices for the implicit time-integration method.

        A_lhs @ u_new = A_rhs @ u_old + A_reaction @ reaction_term

        A_lhs = M_lumped + theta * dt * K
        A_rhs = M_lumped - (1 - theta) * dt * K
        A_reaction = dt * M_reaction_lumped       if reaction_lumping
        A_reaction = dt * M                       otherwise
        """
        dt = self.simulation.dt
        theta = 0.5 if self.order == 2 else 1.0
        myo_indexes = self.simulation.cardiac_tissue.myo_indexes

        stiff, mass = self.simulation.spatial_discretization.weights
        mass_lumped = self.assemble_lumped_mass_matrix(mass)
        a_lhs_matrix = self.assemble_lhs_matrix(stiff, mass_lumped, dt, theta)
        a_rhs_matrix = self.assemble_rhs_matrix(stiff, mass_lumped, dt, theta)
        a_reaction_matrix = self.assemble_reaction_matrix(mass, dt)

        self.a_lhs_matrix = self.simulation.backend.wrap_sparse(
            a_lhs_matrix, indexes=myo_indexes, local_indexing=True)
        self.a_rhs_matrix = self.simulation.backend.wrap_sparse(
            a_rhs_matrix, indexes=myo_indexes, row_reduced=True)
        self.a_reaction_matrix = self.simulation.backend.wrap_sparse(
            a_reaction_matrix, indexes=myo_indexes, row_reduced=True)

        self.myo_indexes = self.simulation.backend.wrap_indexes(myo_indexes)

    def assemble_lumped_mass_matrix(self, mass):
        """Assembles the lumped mass matrix for the Implicit method.

        Parameters
        ----------
        mass : scipy.sparse.csr_matrix
            The mass matrix from the spatial discretization.

        Returns
        -------
        scipy.sparse.csr_matrix
            The lumped mass matrix.
        """
        mass_lumped = ((1 - self.lumping_factor) * mass +
                       self.lumping_factor * sp.diags(mass.sum(axis=1).A1))
        return mass_lumped

    def assemble_rhs_matrix(self, stiff, mass, dt, theta):
        """Assembles the right-hand side matrix for the Implicit method.

        Parameters
        ----------
        stiff : scipy.sparse.csr_matrix
            The stiffness matrix from the spatial discretization.
        mass : scipy.sparse.csr_matrix
            The mass matrix from the spatial discretization.
        dt : float
            Time step for the simulation.
        theta : float
            Weight of the implicit diffusion contribution.

        Returns
        -------
        scipy.sparse.csr_matrix
            The right-hand side system matrix.
        """
        return mass - (1 - theta) * dt * stiff

    def assemble_lhs_matrix(self, stiff, mass, dt, theta):
        """Assembles the left-hand side matrix for the Implicit method.

        Parameters
        ----------
        stiff : scipy.sparse.csr_matrix
            The stiffness matrix from the spatial discretization.
        mass : scipy.sparse.csr_matrix
            The mass matrix from the spatial discretization.
        dt : float
            Time step for the simulation.
        theta : float
            Weight of the implicit diffusion contribution.

        Returns
        -------
        scipy.sparse.csr_matrix
            The left-hand side system matrix.
        """
        return mass + theta * dt * stiff

    def assemble_reaction_matrix(self, mass, dt):
        """Assembles the reaction matrix for the implicit method.

        Parameters
        ----------
        mass : scipy.sparse.csr_matrix
            The mass matrix from the spatial discretization.
        dt : float
            Time step for the simulation.

        Returns
        -------
        scipy.sparse.csr_matrix
            The reaction matrix. If ``reaction_lumping`` is True, returns the
            time-scaled lumped mass matrix; otherwise, returns the time-scaled
            consistent mass matrix.
        """
        if self.reaction_lumping:
            return dt * sp.diags(mass.sum(axis=1).A1)

        return dt * mass

    def run(self):
        """Performs one implicit time-integration step.

        For each time step:
            1. Update the solution vector and reaction term from the cardiac
               model.
            2. Swap references for in-place updates of the solution vectors.
            3. Form the right-hand side from ``A_rhs @ u_old`` and
               ``A_reaction @ reaction_term``.
            4. Solve ``A_lhs @ u_new = b`` with the configured linear solver.
            5. Update the cardiac model solution with the new values.
        """
        self.u = self.simulation.cardiac_model.u
        self.reaction_term = self.simulation.cardiac_model.rhs
        self.u_old, self.u_old_2, self.u = self.u, self.u_old, self.u_old_2

        x0, b = self.linalg.prepare_implicit_step(
            self.a_rhs_matrix, self.a_reaction_matrix, self.u_old, self.u_old_2,
            self.reaction_term, self.myo_indexes
        )
        u_new, n_iter = self.solver(
            self.a_lhs_matrix, b, x0, atol=self.atol, maxiter=self.maxiter
        )

        self.u = self.linalg.update_at_active_indexes(u_new, self.myo_indexes, self.u)

        if n_iter < 0:
            warnings.warn("Diffusion kernel solution accuracy is not reached")

        self.num_iterations.append(n_iter)
        self.simulation.cardiac_model.u = self.u
        return self.u
