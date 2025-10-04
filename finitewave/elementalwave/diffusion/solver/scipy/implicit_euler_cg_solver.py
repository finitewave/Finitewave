from scipy.sparse import linalg
import warnings
from threadpoolctl import threadpool_limits

from finitewave.elementalwave.diffusion.solver.solver import Solver


class ImplicitEulerCGSolver(Solver):
    def __init__(self):
        self.maxiter = None
        self.rtol = 1e-6
        self.atol = 0
        self.preconditioner = None

    def assemble_system(self, stiffness_matrix, mass_matrix, dt):
        a_matrix = mass_matrix + dt * stiffness_matrix
        return [a_matrix, mass_matrix]

    @threadpool_limits.wrap(limits=1, user_api="blas")
    def solve(self, u, rhs, matrices):
        a_matrix = matrices[0]
        mass_matrix = matrices[1]
        b = mass_matrix @ (u + rhs)
        u, success = linalg.cg(a_matrix, b, x0=u, atol=self.atol,
                               rtol=self.rtol, maxiter=self.maxiter,
                               M=self.preconditioner)
        if success > 0:
            warnings.warn("Diffusion kernel solution accuracy is not reached")
        return u
