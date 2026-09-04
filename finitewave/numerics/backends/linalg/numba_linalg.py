import numpy as np
from numba import njit, prange


def select_explicit_solver(x, active_indexes):
    if x.size >= active_indexes.size:
        return explicit_step
    
    raise ValueError("Invalid combination of x and active_indexes sizes.")


def explicit_step(A_x, x, A_y, y, active_mask, out):
    """Performs the matrix-vector multiplication and addition with half-lumped mass matrix.

    Parameters
    ----------
    A_x : tuple
        The stiffness matrix representing the diffusion operator.
    x : np.ndarray
        Matrix multiplier input vector.
    A_y : tuple
        The mass matrix representing the mass operator.
    y : np.ndarray
        Vector to be scaled and added to the result of the matrix-vector multiplication.
    active_mask : np.ndarray
        Boolean mask indicating the active cells in the simulation.
    out : np.ndarray
        Output solution vector to store the updated solution after the Forward Euler step.

    Returns
    -------
    np.ndarray
        Output solution vector to store the updated solution after the Forward Euler step.
    """
    out = matvec_numba(A_x[0], A_x[1], A_x[2], x, out)
    return matvec_p_ay_numba(A_y[0], A_y[1], A_y[2], y, 1.0, out, out)


def prepare_implicit_step(A_rhs, A_ion, x_old, x_old_2, i_ion, active_indexes):
    x0 = np.empty(x_old.size, dtype=x_old.dtype)
    x0 = ax_p_by_numba(2.0, x_old, -1.0, x_old_2, x0)

    b = np.empty(x_old.size, dtype=x_old.dtype)
    b = matvec_numba(A_ion[0], A_ion[1], A_ion[2], i_ion, b)
    b = matvec_p_ay_numba(A_rhs[0], A_rhs[1], A_rhs[2], x_old, 1.0, b, b)

    if active_indexes.size != x_old.size:
        x0 = select_values_numba(x0, active_indexes)
        b = select_values_numba(b, active_indexes)

    return x0, b

def update_at_active_indexes(x, active_indexes, out):
    if active_indexes.size != out.size:
        out = set_values_numba(x, active_indexes, out)
    else:
        out = x
    return out


def cg(A, b, x0, M=None, atol=1e-6, maxiter=100):
    """ Conjugate Gradient solver for Ax = b for x, where A is a sparse
    matrix given in CSR format.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The matrix A in Ax = b.
    b : 1D array of float
        Right-hand side vector.
    x0 : 1D array of float
        Initial guess for the solution, will be modified in place.
    M : scipy.sparse.csr_matrix, optional
        Preconditioner matrix.
    atol : float
        Absolute tolerance for the stopping criterion.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    x : 1D array of float
        Approximate solution vector.
    int
        Number of iterations performed, or -1 if not converged within maxiter.
    """
    indptr, indices, data = A

    b_norm_2 = dot_numba(b, b)

    if b_norm_2 == 0: 
        return x0, 0

    # if rtol is not None:
    #     atol = max(atol, rtol * np.sqrt(b_norm_2))

    r = np.empty_like(x0)
    q = np.empty_like(x0)
    p = np.empty_like(x0)

    r = matvec_numba(indptr, indices, data, x0, r)
    r = ax_p_by_numba(1.0, b, -1.0, r, r)

    p = copyto_numba(r, p)
    r_norm = dot_numba(r, r)

    if np.sqrt(r_norm) < atol:
        return x0, 0

    x = x0

    for iteration in range(maxiter):

        q = matvec_numba(indptr, indices, data, p, q)
        p_dot_q = dot_numba(p, q)

        alpha = r_norm / p_dot_q

        x = ax_p_by_numba(1.0, x, alpha, p, x)
        r = ax_p_by_numba(1.0, r, -alpha, q, r)
        r_norm_new = dot_numba(r, r)

        if np.sqrt(r_norm_new) < atol:
            return x, iteration

        beta = r_norm_new / r_norm
        p = ax_p_by_numba(beta, p, 1.0, r, p)
        r_norm = r_norm_new

    return x, iteration


@njit(parallel=True, fastmath=True, cache=True)
def matvec_p_ay_numba(indptr, indices, data, x, a, y, out):
    """
    Computes out = A @ x + a * y for a sparse matrix A in CSR format.

    Note: The A should be in global indexing
    """
    n_rows = indptr.shape[0] - 1

    for i in prange(n_rows):
        start, end = indptr[i], indptr[i+1]
        if start == end:
            continue
        a_x = 0.0
        for j in range(start, end):
            jj = indices[j]
            a_x += data[j] * x.flat[jj]

        out.flat[i] = a_x + a * y.flat[i]

    return out


@njit(parallel=True, fastmath=True, cache=True)
def matvec_numba(indptr, indices, data, x, out):
    """
    Computes out = A @ x for a sparse matrix A in CSR format.
    """
    n = indptr.shape[0] - 1
    for i in prange(n):
        start, end = indptr[i], indptr[i+1]
        if start == end:
            continue
        out_i = 0.
        for j in range(start, end):
            jj = indices[j]
            out_i += data[j] * x.flat[jj]

        out.flat[i] = out_i
    return out


@njit(parallel=True, fastmath=True, cache=True)
def dot_numba(x, y):
    return np.dot(x, y)


@njit(parallel=True, fastmath=True, cache=True)
def ax_p_by_numba(a, x, b, y, out):
    """Computes out = a * x + b * y.
    """
    n = x.size
    for i in prange(n):
        out.flat[i] = a * x.flat[i] + b * y.flat[i]
    return out


@njit(parallel=True, fastmath=True, cache=True)
def copyto_numba(x, out):
    """
    Copies x to y in place for the given indexes.
    """
    n = x.size
    for i in prange(n):
        out.flat[i] = x.flat[i]
    return out

@njit(parallel=True, fastmath=True, cache=True)
def select_values_numba(arr, inds):
    """
    Selects values from arr at the specified indices.

    Parameters
    ----------
    arr : np.ndarray
        Input array from which values are to be selected.
    inds : np.ndarray
        Indices of the values to be selected.

    Returns
    -------
    np.ndarray
        Array containing the selected values.
    """
    out = np.empty(inds.shape, dtype=arr.dtype)
    for i in prange(inds.size):
        out[i] = arr[inds[i]]
    return out


@njit(parallel=True, fastmath=True, cache=True)
def set_values_numba(values, inds, arr):
    """
    Sets values in arr at the specified indices.

    Parameters
    ----------
    arr : np.ndarray
        Input array in which values are to be set.
    inds : np.ndarray
        Indices at which values are to be set.
    values : np.ndarray
        Values to be set at the specified indices.

    Returns
    -------
    np.ndarray
        Array with updated values.
    """
    for i in prange(inds.size):
        arr[inds[i]] = values[i]
    return arr
