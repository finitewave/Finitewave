import warnings
import numpy as np
from scipy.sparse import linalg
from numba import njit, prange


class NumbaCGSolver:
    def __init__(self):
        self.maxiter = None
        self.rtol = 1e-5
        self.atol = 1e-6
        self.preconditioner = None

    def axpy(self, a, b, dt):
        return a + dt * b

    def axmy(self, a, b, dt):
        return a - dt * b

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


# @njit
def csr_matvec(data, indices, indptr, x, y):
    n = indptr.shape[0] - 1
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start, end = indptr[i], indptr[i+1]
        for j in range(start, end):
            y[i] += data[j] * x[indices[j]]
    return y


# @njit(parallel=True)
def cg_sparse(data, indices, indptr, b, tol=1e-8, maxiter=1000):
    n = len(b)
    x = np.zeros(n, dtype=np.float64)
    Ap = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    r = b - csr_matvec(data, indices, indptr, x, r)
    p = r.copy()
    rsold = r @ r

    for k in range(maxiter):
        Ap = csr_matvec(data, indices, indptr, p, Ap)
        alpha = rsold / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r @ r
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x, k + 1
