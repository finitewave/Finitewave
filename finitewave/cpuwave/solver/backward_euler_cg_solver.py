
from .crank_nicolson_cg_solver import CrankNicolsonCGSolver


class BackwardEulerCGSolver(CrankNicolsonCGSolver):
    """Implements the Backward Euler implicit time integration method
    with Conjugate Gradient solver for implicit diffusion step.

    Attributes
    ----------
    Inherits all attributes from CrankNicolsonCGSolver.
    """
    def __init__(self):
        super().__init__()

    def assemble_system(self):
        """Assembles the system matrix for the Backward Euler method.

        A_lhs = M + dt * K
        A_rhs = M
        """
        stiff, mass = self.simulation.diffusion_model.weights
        dt = self.simulation.dt
        self.a_lhs_matrix = mass + dt * stiff
        self.a_rhs_matrix = mass
