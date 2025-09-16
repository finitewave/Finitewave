from scipy.sparse import linalg
import warnings
from threadpoolctl import threadpool_limits


class ScipyCGSolver:
    def __init__(self):
        self.maxiter = None
        self.rtol = 1e-5
        self.atol = 1e-6
        self.preconditioner = None

    def axpy(self, a, b, dt):
        return a + dt * b

    def axmy(self, a, b, dt):
        return a - dt * b

    @threadpool_limits.wrap(limits=1, user_api="blas")
    def diffusion_kernel(self, u_new, u, rhs, matrices):
        a_matrix = matrices[0]
        mass_matrix = matrices[1]
        b = mass_matrix.dot(u + rhs)
        u_new, success = linalg.cg(a_matrix, b, x0=u, atol=self.atol,
                                   rtol=self.rtol, maxiter=self.maxiter,
                                   M=self.preconditioner)
        if success > 0:
            warnings.warn("Diffusion kernel solution accuracy is not reached")
        return u_new
