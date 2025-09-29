from scipy.sparse import linalg
import warnings
from threadpoolctl import threadpool_limits

from finitewave.femwave.diffusion.solver.solver import Solver


class CrankNicolsonCGSolver(Solver):
    def __init__(self):
        self.maxiter = None
        self.rtol = 1e-6
        self.atol = 0
        self.preconditioner = None

    def assemble_system(self, stiffness_matrix, mass_matrix, dt):
        a_lhs_matrix = mass_matrix + 0.5 * dt * stiffness_matrix
        a_rhs_matrix = mass_matrix - 0.5 * dt * stiffness_matrix
        return [a_lhs_matrix, a_rhs_matrix, mass_matrix]

    @threadpool_limits.wrap(limits=1, user_api="blas")
    def solve(self, u, rhs, matrices):
        a_lhs_matrix = matrices[0]
        a_rhs_matrix = matrices[1]
        mass_matrix = matrices[2]
        b = a_rhs_matrix @ u + mass_matrix @ rhs
        u, success = linalg.cg(a_lhs_matrix, b, x0=u, atol=self.atol,
                               rtol=self.rtol, maxiter=self.maxiter,
                               M=self.preconditioner)
        if success > 0:
            warnings.warn("Diffusion kernel solution accuracy is not reached")
        return u
