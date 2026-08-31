import numpy as np
from numba import njit, prange


def explicit_step(A, x, a, y, active_indexes, out):
    """Performs the matrix-vector multiplication and addition.

    Parameters
    ----------
    indptr : np.ndarray
        The index pointer array of the CRS format.
    indices : np.ndarray
        The column indices of the non-zero elements in CRS format.
    data : np.ndarray
        The non-zero values of the matrix in CRS format.
        The stiffness matrix representing the diffusion operator.
    x : np.ndarray
        Matrix multiplier input vector.
    y : np.ndarray
        Vector to be scaled and added to the result of the matrix-vector multiplication.
    a : float
        Scaling factor for the y vector.
    active_indexes : np.ndarray
        Array of indexes where the solution is defined.
    out : np.ndarray
        Output solution vector to store the updated solution after the Forward Euler step.

    Returns
    -------
    np.ndarray
        Updated solution vector after the Forward Euler step.
    """
    indptr, indices, data = A
    return matvec_p_ay_numba(indptr, indices, data, x, a, y, active_indexes, out)


def explicit_step_half_lumped(A_x, x, A_y, y, active_indexes, out):
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
    active_indexes : np.ndarray
        Array of indexes where the solution is defined.
    out : np.ndarray
        Output solution vector to store the updated solution after the Forward Euler step.

    Returns
    -------
    np.ndarray
        Output solution vector to store the updated solution after the Forward Euler step.
    """
    indptr_x, indices_x, data_x = A_x
    indptr_y, indices_y, data_y = A_y
    out = matvec_numba(indptr_x, indices_x, data_x, x, active_indexes, out)
    return matvec_p_ay_numba(indptr_y, indices_y, data_y, y, 1.0, out, active_indexes, out)


def implicit_step(method, A_lhs, A_rhs, A_ion, x_old, x_old_2, i_ion, active_indexes, out, **kwargs):
    """Performs the implicit time-stepping operation.

    Parameters
    ----------
    method : str
        The time-stepping method to be used.
    A_lhs : tuple
        The left-hand side matrix in CRS format.
    A_rhs : tuple
        The right-hand side matrix in CRS format.
    A_ion : tuple
        The ionic current matrix in CRS format.
    x_old : np.ndarray
        The solution vector from the previous time step.
    x_old_2 : np.ndarray
        The solution vector from two time steps ago (used for certain methods).
    i_ion : np.ndarray
        The ionic current vector.
    active_indexes : np.ndarray
        Array of indexes where the solution is defined.
    out : np.ndarray
        Output solution vector to store the updated solution after the implicit step.

    Returns
    -------
    np.ndarray
        Updated solution vector after the implicit step.
    """
    indptr_rhs, indices_rhs, data_rhs = A_rhs
    indptr_ion, indices_ion, data_ion = A_ion

    x_old_2 = ax_p_by_numba(2.0, x_old, -1.0, x_old_2, active_indexes, out)
    # Compute the right-hand side vector
    rhs = matvec_numba(indptr_rhs, indices_rhs, data_rhs, x_old, active_indexes, out)
    rhs = matvec_p_ay_numba(indptr_ion, indices_ion, data_ion, i_ion, 1.0, rhs, active_indexes, out)

    # Solve the linear system using Conjugate Gradient method
    x_new, _ = method(A_lhs, rhs, x_old_2, atol=kwargs.get('atol', 1e-6), maxiter=kwargs.get('maxiter', 100))

    return x_new


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
    indexes : 1D array of int
        Array of indexes where the solution is defined.
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
    indptr, indices, data, indexes = A

    b_norm_2 = dot_numba(b, b, indexes)

    if b_norm_2 == 0:
        return x0, 0

    # if rtol is not None:
    #     atol = max(atol, rtol * np.sqrt(b_norm_2))

    r = np.empty_like(x0)
    r = b_m_matvec_numba(indptr, indices, data, x0, b, indexes, r)
    q = np.empty_like(r)

    p = np.empty_like(r)
    p = copyto_numba(r, indexes, p)

    r_norm = dot_numba(r, r, indexes)

    if np.sqrt(r_norm) < atol:
        return x0, 0

    for iteration in range(maxiter):

        q, p_dot_q = matvec_and_dot_numba(indptr, indices, data, p, indexes, q)
        alpha = r_norm / p_dot_q

        x, r, r_norm_new = y_pm_ax_numba(alpha, p, x0, q, r, indexes)

        if np.sqrt(r_norm_new) < atol:
            return x, iteration

        beta = r_norm_new / r_norm
        p = ax_p_by_numba(beta, p, 1.0, r, indexes, p)
        r_norm = r_norm_new

    return x, iteration


@njit(parallel=True, fastmath=True, cache=True)
def matvec_p_ay_numba(indptr, indices, data, x, a, y, indexes, out):
    """
    Computes out = A @ x + a * y for a sparse matrix A in CSR format.

    Note: The A should be in global indexing
    """
    n_rows = indexes.shape[0]

    for ii in prange(n_rows):
        i = indexes[ii]
        start, end = indptr[ii], indptr[ii+1]
        if start == end:
            continue
        a_x = 0.0
        for j in range(start, end):
            jj = indices[j]
            jj = indexes[jj]
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
def matvec_and_dot_numba(indptr, indices, data, x, out):
    """Computes y = A @ x and returns the dot product x^T @ (A @ x).
    """
    n = indptr.shape[0] - 1
    s = 0.
    for i in prange(n):
        start, end = indptr[i], indptr[i+1]
        if start == end:
            continue
        out_i = 0.
        for j in range(start, end):
            jj = indices[j]
            out_i += data[j] * x.flat[jj]

        out.flat[i] = out_i
        s += out_i * x.flat[i]
    return out, s


@njit(parallel=True, fastmath=True, cache=True)
def b_m_matvec_numba(indptr, indices, data, b, x, out):
    """
    Computes y = b - A @ x
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

        out.flat[i] = b.flat[i] - out_i
    return out


@njit(parallel=True, fastmath=True, cache=True)
def dot_numba(x, y):
    return np.dot(x, y)

@njit(parallel=True, fastmath=True, cache=True)
def ax_p_by_numba(a, x, b, y, indexes, out):
    """Computes out = a * x + b * y.
    """
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        out.flat[ii] = a * x.flat[ii] + b * y.flat[ii]
    return out


@njit(parallel=True, fastmath=True, cache=True)
def y_pm_ax_numba(a, xp, yp, xm, ym, indexes):
    """
    Computes yp = yp + a * xp and ym = ym - a * xm in place.
    """
    ym_dot_ym = 0.
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        yp.flat[ii] += a * xp.flat[ii]

        ym_i = ym.flat[ii] - a * xm.flat[ii]
        ym.flat[ii] = ym_i
        ym_dot_ym += ym_i * ym_i
    return yp, ym, ym_dot_ym


@njit(parallel=True, fastmath=True, cache=True)
def copyto_numba(x, indexes, out):
    """
    Copies x to y in place for the given indexes.
    """
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        out.flat[ii] = x.flat[ii]
    return out
    