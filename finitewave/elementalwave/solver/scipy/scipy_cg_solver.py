from scipy.sparse import linalg
import warnings


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

    def diffusion_kernel(self, u_new, u, rhs, matrices, indexes):
        a_matrix = matrices[0]
        mass_matrix = matrices[1]
        b = mass_matrix.dot(u[indexes] + rhs[indexes])
        u_new[indexes], success = linalg.cg(a_matrix,
                                            b,
                                            x0=u[indexes],
                                            atol=self.atol,
                                            rtol=self.rtol,
                                            maxiter=self.maxiter,
                                            M=self.preconditioner)
        if success > 0:
            warnings.warn("Diffusion kernel solution accuracy is not reached")
