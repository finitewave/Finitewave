import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from .numba_linalg import cg_numba


def poisson_cg_solver(A, b, indexes, dirichlet_indexes=None, x0=None,
                      rtol=None, atol=1e-8, maxiter=1000):
    """
    Solves the linear system Ax = b using the Conjugate Gradient method,
    with support for Dirichlet boundary conditions.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The sparse matrix representing the linear system.
    b : np.ndarray
        The right-hand side vector.
    indexes : np.ndarray
        The indexes of the nodes where the solution is computed
        (including dirichlet nodes).
    dirichlet_indexes : np.ndarray, optional
        The indexes of the nodes where Dirichlet boundary conditions are applied.
    x0 : np.ndarray, optional
        The initial guess for the solution vector.
        If None, it will be set to a copy of b.
    atol : float, optional
        The absolute tolerance for convergence. Default is 1e-8.
    maxiter : int, optional
        The maximum number of iterations for the Conjugate Gradient solver.
        Default is 1000.

    Returns
    -------
    np.ndarray
        The solution vector x.
    int
        An integer flag indicating whether the solver converged (>0) or
        not (<0).
    """

    if x0 is None:
        x0 = b.copy()

    if dirichlet_indexes is not None:
        internal_idx = internal_indexes(A, dirichlet_indexes)
        A, b = apply_dirichlet_bc(A, b, x0, internal_idx, dirichlet_indexes)
        indexes = indexes[internal_idx]

    x, success = cg_numba(A.indptr, A.indices, A.data, b, x0, indexes,
                          rtol=rtol, atol=atol, maxiter=maxiter)
    return x, success
    

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


def apply_dirichlet_bc(matrix, b, x0, internal_idx, dirichlet_idx):
    """
    Applies Dirichlet boundary conditions to the given matrix by removing
    rows and columns corresponding to the specified indexes.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        The matrix to which Dirichlet boundary conditions will be applied.
    b : np.ndarray
        The right-hand side vector.
    x0 : np.ndarray
        The initial guess for the solution vector.
    internal_idx : np.ndarray
        The indexes of the internal nodes, where the solution will be
        computed.
    dirichlet_idx : np.ndarray
        The indexes of the nodes where Dirichlet boundary conditions
        are applied.

    Returns
    -------
    scipy.sparse.csr_matrix
        The modified matrix with Dirichlet boundary conditions applied.
    scipy.sparse.csr_matrix
        The boundary matrix corresponding to Dirichlet nodes.
    """
    internal_matrix = matrix[internal_idx][:, internal_idx]
    boundary_matrix = matrix[internal_idx][:, dirichlet_idx]
    b[internal_idx] -= boundary_matrix @ x0[dirichlet_idx]
    return internal_matrix, b
