from scipy import sparse
from finitewave.core.solver.solver_base import SolverBase


class ForwardEulerSolver(SolverBase):
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
    myo_indexes : np.ndarray
        Indexes of myocardial nodes in the simulation.
    num_iterations : list
        List to track the number of iterations per time step.
    """
    def __init__(self, full_lumping=True):
        self.full_lumping = full_lumping
        self.a_matrix = None
        self.u_new = None
        self.u = None
        self.rhs = None
        self.myo_indexes = None
        self.num_iterations = []
        self.linalg_method = None

    def initialize(self, simulation):
        """Initializes the Forward Euler solver with the given simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object containing the cardiac and diffusion models.
        """
        self.simulation = simulation
        self.linalg = simulation.backend.linalg

        if self._solve is None:
            self._solve = getattr(self.linalg, self.method)

        self.num_iterations = []
        self.u = simulation.cardiac_model.u
        self.u_old = simulation.backend.copy(self.u)
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
        dt = self.simulation.dt
        self.myo_indexes = self.simulation.cardiac_model.myo_indexes

        stiff, mass = self.simulation.diffusion_model.weights
        mass_inv = sparse.diags(1 / mass.sum(axis=1).A1)

        a_rhs_matrix = sparse.eye(stiff.shape[0]) - dt * mass_inv * stiff
        self.a_rhs_matrix = self.simulation.backend.wrap_sparse(a_rhs_matrix, self.myo_indexes)

        if not self.full_lumping:
            a_ion_matrix = dt * mass_inv @ mass
            self.a_ion = self.simulation.backend.wrap_sparse(a_ion_matrix, self.myo_indexes)
            self._solver = self._double_loop_solver

        else:
            self.a_ion = dt
            self._solver = self._single_loop_solver


    def run(self):
        """Performs a single time step using the Forward Euler method.

        For each time step:
            1. Update the solution vector and right-hand side from the cardiac model.
            2. u_new = (I - dt * M^{-1} * K) @ u + dt * rhs (explicit diffusion step).
            3. Update the cardiac model solution with the new values.
        """        
        self.u = self.simulation.cardiac_model.u
        self.rhs = self.simulation.cardiac_model.rhs
        self.myo_indexes = self.simulation.cardiac_model.myo_indexes

        self.u_old, self.u = self.u, self.u_old

        self.u = self._solver(self.a_rhs_matrix, self.a_ion, self.u_old, self.rhs, self.u)

        self.simulation.cardiac_model.u = self.u        
        self.num_iterations.append(1)

    def _single_loop_solver(self, a_rhs, a_ion, u_old, i_ion, u):
        """Performs a step of the Forward Euler method in single loop.

        u = A_rhs @ u_old + a_ion * i_ion

        Parameters
        ----------
        a_rhs : scipy.sparse.csr_matrix
            The system matrix for the diffusion step.
        a_ion : float
            The scalar multiplier for the ionic current.
        u_old : np.ndarray
            The solution vector at the previous time step.
        i_ion : np.ndarray
            The ionic current vector.
        u : np.ndarray
            The output vector to store the new solution.

        Returns
        -------
        np.ndarray
            The updated solution vector after the Forward Euler step.
        """
        return self.linalg.matvec_p_ay(a_rhs, u_old, i_ion, a_ion, u)

    def _double_loop_solver(self, a_rhs, a_ion, u_old, i_ion, u):
        """Performs a step of the Forward Euler method in double loop.

        u_ion = A_ion @ i_ion
        u = A_rhs @ u_old + u_ion

        Parameters
        ----------
        a_rhs : scipy.sparse.csr_matrix
            The system matrix for the diffusion step.
        a_ion : scipy.sparse.csr_matrix
            The system matrix for the ionic current step.
        u_old : np.ndarray
            The solution vector at the previous time step.
        i_ion : np.ndarray
            The ionic current vector.
        u : np.ndarray
            The output vector to store the new solution.

        Returns
        -------
        np.ndarray
            The updated solution vector after the Forward Euler step.
        """
        u_ion = self.linalg.matvec(a_ion, i_ion, u)
        return self.linalg.matvec_p_ay(a_rhs, u_old, u_ion, 1., u)

