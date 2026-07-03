
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
        dt = self.simulation.dt
        dtype = self.simulation.backend.dtype

        stiff, mass = self.simulation.diffusion_model.weights
        a_lhs_matrix = mass + dt * stiff
        a_rhs_matrix = mass

        self.a_lhs_matrix = self.linalg_method.wrap_matrix(a_lhs_matrix, dtype)
        self.a_rhs_matrix = self.linalg_method.wrap_matrix(a_rhs_matrix, dtype)
