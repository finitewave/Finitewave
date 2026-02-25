import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from .numba_linalg import cg_numba


class PoissonCGSolver:
    def __init__(self):
        pass

    def run(self, A, b, indexes, dirichlet_indexes, x0=None, tol=1e-8,
            max_iter=1000):

        if x0 is None:
            x0 = b.copy()

        if dirichlet_indexes is not None:
            A, b = self.apply_dirichlet_bc(A, indexes, dirichlet_indexes, b)

        x, success = cg_numba(A.indptr, A.indices, A.data, b, x0 if x0 is not None else b.copy(),
                              indexes, atol=tol, maxiter=max_iter)
        

    def internal_indexes(matrix, dirichlet_idx):
        """
        Computes the indexes of the internal nodes by excluding the Dirichlet
        boundary nodes.

        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The matrix to which Dirichlet boundary conditions will be applied.
        dirichlet_idx : np.ndarray
            The indexes of the nodes where Dirichlet boundary conditions are applied.

        Returns
        -------
        np.ndarray
            The indexes of the internal nodes.
        """
        mask = np.ones(matrix.shape[0], dtype=bool)
        mask[dirichlet_idx] = False
        internal_idx = np.where(mask)[0]
        return internal_idx

    def apply_dirichlet_bc(self, matrix, b, x0, internal_idx, dirichlet_idx):
        """
        Applies Dirichlet boundary conditions to the given matrix by removing
        rows and columns corresponding to the specified indexes.

        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The matrix to which Dirichlet boundary conditions will be applied.
        dirichlet_idx : np.ndarray
            The indexes of the nodes where Dirichlet boundary conditions are applied.

        Returns
        -------
        scipy.sparse.csr_matrix
            The modified matrix with Dirichlet boundary conditions applied.
        scipy.sparse.csr_matrix
            The boundary matrix corresponding to Dirichlet nodes.
        """
        internal_matrix = matrix[internal_idx][:, internal_idx]
        bounary_matrix = matrix[internal_idx][:, dirichlet_idx]
        b = b - bounary_matrix @ x0[dirichlet_idx]
        return internal_matrix, bounary_matrix

    @threadpool_limits.wrap(limits=1, user_api="blas")
    def laplace_solver(K, dirichlet_idx, x0, atol=1e-8, maxiter=1000, **kwargs):
        """
        Solves the linear system Ax = b using the Conjugate Gradient method from SciPy.

        Parameters
        ----------
        K : scipy.sparse.csr_matrix
            The stiffness matrix.
        dirichlet_idx : np.ndarray
            The indexes of the nodes where Dirichlet boundary conditions are applied.
        x0 : np.ndarray
            The initial guess for the solution vector.
        atol : float, optional
            The absolute tolerance for convergence (default is 1e-8).
        maxiter : int, optional
            The maximum number of iterations (default is 1000).

        Returns
        -------
        x : numpy.ndarray
            The solution vector.
        info : int
            Convergence information (0 if successful).
        """
        internal_idx = internal_indexes(K, dirichlet_idx)
        K_internal, K_boundary = apply_dirichlet_bc(K, internal_idx, dirichlet_idx)
        b = b - K_boundary @ x0[dirichlet_idx]
        x0_ = x0[internal_idx].copy()

        x, info = sparse.linalg.cg(K_internal, b, x0=x0_, atol=atol, 
                                maxiter=maxiter, **kwargs)
        x0[internal_idx] = x
        return x0, info