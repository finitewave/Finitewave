import numpy as np
from numba import njit, prange


class JacobiPreconditioner:
    def __init__(self, A):
        self.jacobi = None
        self.build(A)

    def matvec(self, x, y, indexes):
        return jacobi_preconditioner(self.jacobi, x, y, indexes)
    
    def build(self, A):
        A_diag = A.diagonal()
        self.jacobi = np.ones_like(A_diag)
        self.jacobi[A_diag != 0] = 1. / A_diag[A_diag != 0]


@njit(parallel=True, fastmath=True, cache=True)
def jacobi_preconditioner(jacobi, x, y, indexes):
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        y.flat[ii] = jacobi.flat[ii] * x.flat[ii]
    return y