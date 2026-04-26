
from .crank_nicolson_solver import CrankNicolsonSolver


class BackwardEulerSolver(CrankNicolsonSolver):
    """Implements the Backward Euler implicit time integration method
    with Conjugate Gradient solver for implicit diffusion step.

    Attributes
    ----------
    Inherits all attributes from CrankNicolsonCGSolver.
    """
    def __init__(self, atol=1e-8, maxiter=100):
        super().__init__(atol=atol, maxiter=maxiter)

    def assemble_system(self):
        """Assembles the system matrix for the Backward Euler method.

        A_lhs = M + dt * K
        A_rhs = M
        """
        stiff, mass = self.simulation.diffusion_model.weights
        dt = self.simulation.dt
        self.a_lhs_matrix = mass + dt * stiff
        self.a_rhs_matrix = mass

        if self.simulation.backend.sparse_support:
            self.a_lhs_matrix = self.crs_to_numpy(self.a_lhs_matrix)
            self.a_rhs_matrix = self.crs_to_numpy(self.a_rhs_matrix)

        else:
            self.a_lhs_matrix = self.csr_to_ellpack(self.a_lhs_matrix)
            self.a_rhs_matrix = self.csr_to_ellpack(self.a_rhs_matrix)
