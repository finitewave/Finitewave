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
                 full_lumping=False, method="cg"):
        self.atol = atol
        self.maxiter = maxiter
        self.lumping_factor = lumping_factor
        self.order = order
        self.full_lumping = full_lumping
        self.method = method
        
        self.b = None
        self.u = None
        self.u_old = None
        self.u_old_2 = None

        self.a_lhs_matrix = None
        self.a_rhs_matrix = None
        self.a_ion_matrix = None

        self.linalg = None
        self._solve = None

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
        self._solve = getattr(self.linalg, self.method)

        if self.order not in [1, 2]:
            raise ValueError("ImplicitSolver order must be 1 or 2.")

        self.num_iterations = []
        self.u = simulation.cardiac_model.u
        self.b = simulation.backend.wrap_array(0. * self.u)
        self.u_old = simulation.backend.copy(self.u)
        self.u_old_2 = simulation.backend.copy(self.u)
        self.rhs = simulation.cardiac_model.rhs
        self.myo_indexes = simulation.cardiac_model.myo_indexes
        self.fibro_mask = simulation.cardiac_model.fibro_mask
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

        stiff, mass = self.simulation.diffusion_model.weights
        mass_lumped = ((1 - self.lumping_factor) * mass + 
                       self.lumping_factor * sp.diags(mass.sum(axis=1).A1))
        a_lhs_matrix = mass_lumped + theta * dt * stiff
        a_rhs_matrix = mass_lumped - (1 - theta) * dt * stiff

        if self.full_lumping:
            a_ion_matrix = dt * mass_lumped
        else:
            a_ion_matrix = dt * mass

        self.a_lhs_matrix = self.simulation.backend.wrap_sparse(
            a_lhs_matrix, indexes=self.myo_indexes, local_indexing=True)
        self.a_rhs_matrix = self.simulation.backend.wrap_sparse(
            a_rhs_matrix, indexes=self.myo_indexes)
        self.a_ion_matrix = self.simulation.backend.wrap_sparse(
            a_ion_matrix, indexes=self.myo_indexes)

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
        self.myo_indexes = self.simulation.cardiac_model.myo_indexes
        self.fibro_mask = self.simulation.cardiac_model.fibro_mask

        self.u_old, self.u_old_2, self.u = self.u, self.u_old, self.u_old_2

        # b = A_ion @ rhs
        self.b = self.linalg.matvec(self.a_ion_matrix, self.rhs, self.b)
        # b = A_rhs @ u_old + 1. * b
        self.b = self.linalg.matvec_p_ay(self.a_rhs_matrix, self.u_old, self.b, 1., self.b)
        # Better initial guess for CG solver using the previous two time steps
        self.u = self.linalg.axmy(2., self.u_old, self.u_old_2, self.u)
        self.u, n_iter = self._solve(self.a_lhs_matrix, self.b, self.u, atol=self.atol, maxiter=self.maxiter)

        if n_iter < 0:
            warnings.warn("Diffusion kernel solution accuracy is not reached")
        
        self.num_iterations.append(n_iter)
        self.simulation.cardiac_model.u = self.u
        return self.u
