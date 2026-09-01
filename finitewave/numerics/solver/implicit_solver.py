import warnings
from scipy import sparse as sp
from finitewave.core.solver.solver_base import SolverBase


class ImplicitSolver(SolverBase):
    """Implements the implicit time integration method 
    with Conjugate Gradient solver and half-lumping of the mass matrix
    for cardiac simulations.

    Attributes
    ----------
    maxiter : int
        Maximum number of iterations for the CG solver.
    atol : float
        Absolute tolerance for the CG solver.
    lumping_factor : float
        Factor for mass lumping in the diffusion model. Default is 0 (no lumping).
    order : int
        Order of the implicit method. 1 for Backward Euler, 2 for Crank-Nicolson.
    full_lumping : bool
        If True, uses full lumping of the mass matrix
        (corresponds to operator splitting). Default is False (half-lumping).
    num_iterations : list
        List to track the number of iterations per time step.
    b : np.ndarray
        The right-hand side vector for the linear system.
    u : np.ndarray
        The solution vector at the current time step.
    a_lhs_matrix : scipy.sparse.csr_matrix
        The left-hand side system matrix for the implicit method.
    a_rhs_matrix : scipy.sparse.csr_matrix
        The right-hand side system matrix for the implicit method.
    a_ion_matrix : scipy.sparse.csr_matrix
        The matrix for the ionic model contribution in the implicit method.

    References
    ----------
    .. [1] Pathmanathan, Pras, et al. "Computational modelling of cardiac 
           electrophysiology: explanation of the variability of results 
           from different numerical solvers." 
           International journal for numerical methods in biomedical 
           engineering 28.8 (2012): 890-903.

    """
    def __init__(self, atol=1e-8, maxiter=100, lumping_factor=.0, order=1,
                 ionic_lumping=False):
        self.atol = atol
        self.maxiter = maxiter
        self.lumping_factor = lumping_factor
        self.order = order
        self.ionic_lumping = ionic_lumping
        
        self.u = None
        self.u_old = None
        self.u_old_2 = None

        self.a_lhs_matrix = None
        self.a_rhs_matrix = None
        self.a_ion_matrix = None

        self.linalg = None
        self.solver = None

        self.num_iterations = []

    def initialize(self, simulation):
        """Initializes the Implicit CG solver with the given simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object containing the cardiac and diffusion models.
        """
        self.simulation = simulation

        self.linalg = simulation.backend.linalg

        if self.solver is None:
            self.solver = self.linalg.cg

        if self.order not in [1, 2]:
            raise ValueError("ImplicitSolver order must be 1 or 2.")

        self.num_iterations = []
        self.u = simulation.cardiac_model.u
        self.u_old = simulation.backend.copy(self.u)
        self.u_old_2 = simulation.backend.copy(self.u)
        self.rhs = simulation.cardiac_model.rhs
        self.assemble_system()

    def assemble_system(self):
        """Assembles the system matrix for the Implicit method with
        half-lumping of the mass matrix.

        (Mlumped + dt * K) @ u_new = M_lumped @ u + dt * M @ rhs

        A_lhs = M_lumped + 1 / order * dt * K
        A_rhs = M_lumped + (1 - order) / order * dt * K
        A_ion = dt * M
        """
        dt = self.simulation.dt
        theta = 0.5 if self.order == 2 else 1.0
        myo_indexes = self.simulation.cardiac_tissue.myo_indexes

        stiff, mass = self.simulation.diffusion_model.weights
        mass_lumped = self.assemble_lumped_mass_matrix(mass)
        a_lhs_matrix = self.assemble_lhs_matrix(stiff, mass_lumped, dt, theta)
        a_rhs_matrix = self.assemble_rhs_matrix(stiff, mass_lumped, dt, theta)
        a_ion_matrix = self.assemble_ion_matrix(mass, dt)

        self.a_lhs_matrix = self.simulation.backend.wrap_sparse(
            a_lhs_matrix, indexes=myo_indexes, local_indexing=True)
        self.a_rhs_matrix = self.simulation.backend.wrap_sparse(
            a_rhs_matrix, indexes=myo_indexes, row_reduced=True)
        self.a_ion_matrix = self.simulation.backend.wrap_sparse(
            a_ion_matrix, indexes=myo_indexes, row_reduced=True)

        self.myo_indexes = self.simulation.backend.wrap_indexes(myo_indexes)

    def assemble_lumped_mass_matrix(self, mass):
        """Assembles the lumped mass matrix for the Implicit method.

        Parameters
        ----------
        mass : scipy.sparse.csr_matrix
            The mass matrix from the diffusion model.

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
            The stiffness matrix from the diffusion model.
        mass : scipy.sparse.csr_matrix
            The mass matrix from the diffusion model.
        dt : float
            Time step for the simulation.
        """
        return mass - (1 - theta) * dt * stiff

    def assemble_lhs_matrix(self, stiff, mass, dt, theta):
        """Assembles the left-hand side matrix for the Implicit method.

        Parameters
        ----------
        stiff : scipy.sparse.csr_matrix
            The stiffness matrix from the diffusion model.
        mass : scipy.sparse.csr_matrix
            The mass matrix from the diffusion model.
        dt : float
            Time step for the simulation.
        """
        return mass + theta * dt * stiff

    def assemble_ion_matrix(self, mass, dt):
        """Assembles the ionic current matrix for the Implicit method.

        Parameters
        ----------
        mass : scipy.sparse.csr_matrix
            The mass matrix from the diffusion model.
        dt : float
            Time step for the simulation.
        """
        if self.ionic_lumping:
            return dt * sp.diags(mass.sum(axis=1).A1)
        
        return dt * mass

    def run(self):
        """Performs a single time step using the Implicit method
        with Conjugate Gradient solver for the implicit diffusion step.

        For each time step:
            1. Update the solution vector and right-hand side from the cardiac model.
            2. Swap references for in-place updates of the solution vectors.
            3. Solve the linear system A_lhs @ u_new = A_rhs @ u + A_ion @ rhs using Conjugate Gradient method.
            4. Update the cardiac model solution with the new values.
        """
        self.u = self.simulation.cardiac_model.u
        self.rhs = self.simulation.cardiac_model.rhs
        self.u_old, self.u_old_2, self.u = self.u, self.u_old, self.u_old_2

        x0, b = self.linalg.prepare_implicit_step(
            self.a_rhs_matrix, self.a_ion_matrix, self.u_old, self.u_old_2,
            self.rhs, self.myo_indexes
        )
        u_new, n_iter = self.solver(
            self.a_lhs_matrix, b, x0, atol=self.atol, maxiter=self.maxiter
        )

        self.u = self.linalg.update_active_indexes(u_new, self.myo_indexes, self.u)
        if n_iter < 0:
            warnings.warn("Diffusion kernel solution accuracy is not reached")
        
        self.num_iterations.append(n_iter)
        self.simulation.cardiac_model.u = self.u
        return self.u
