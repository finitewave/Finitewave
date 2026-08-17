import numpy as np
from numba import njit, prange


def axmy(a, x, y, indexes, out):
    """Performs the axmy operation:
    out[indexes] = a * x[indexes] - y[indexes]

    Parameters
    ----------
    a : float
        Scalar multiplier for x.
    x : np.ndarray
        Input vector x.
    y : np.ndarray
        Input vector y.
    indexes : np.ndarray
        Array of indexes where the solution is defined.
    out : np.ndarray
        Output vector to store the result of the axmy operation.

    Returns
    -------
    np.ndarray
        Updated solution vector after the axmy operation.
    """
    return ax_m_y_numba(a, x, y, indexes, out)

def axpy(a, x, y, indexes, out):
    """Performs the axpy operation:
    out[indexes] = a * x[indexes] + y[indexes]

    Parameters
    ----------
    a : float
        Scalar multiplier for x.
    x : np.ndarray
        Input vector x.
    y : np.ndarray
        Input vector y.
    indexes : np.ndarray
        Array of indexes where the solution is defined.
    out : np.ndarray
        Output vector to store the result of the axpy operation.

    Returns
    -------
    np.ndarray
        Updated solution vector after the axpy operation.
    """
    return ax_p_y_numba(a, x, y, indexes, out)

def matvec(A, x, out):
    """Performs the matrix-vector multiplication for a sparse matrix in CRS format.

    Parameters
    ----------
    indptr : np.ndarray
        The index pointer array of the CRS format.
    indices : np.ndarray
        The column indices of the non-zero elements in CRS format.
    data : np.ndarray
        The non-zero values of the matrix in CRS format.
    x : np.ndarray
        Input vector to be multiplied by the matrix.
    indexes : np.ndarray
        Array of indexes where the solution is defined.
    out : np.ndarray
        Output vector to store the result of the matrix-vector multiplication.

    Returns
    -------
    np.ndarray
        Updated solution vector after the matrix-vector multiplication.
    """
    indptr, indices, data, indexes = A
    return matvec_numba(indptr, indices, data, x, out, indexes)

def matvec_p_ay(A, x, y, a, out):
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
    indexes : np.ndarray
        Array of indexes where the solution is defined.
    out : np.ndarray
        Output solution vector to store the updated solution after the Forward Euler step.

    Returns
    -------
    np.ndarray
        Updated solution vector after the Forward Euler step.
    """
    indptr, indices, data, indexes = A
    return matvec_p_ay_numba(indptr, indices, data, x, y, a, indexes, out)


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
    r = b_m_matvec_numba(indptr, indices, data, b, x0, r, indexes)
    q = np.empty_like(r)

    p = np.empty_like(r)
    p = copyto_numba(r, p, indexes)

    r_norm = dot_numba(r, r, indexes)

    if np.sqrt(r_norm) < atol:
        return x0, 0

    for iteration in range(maxiter):

        q, p_dot_q = matvec_and_dot_numba(indptr, indices, data, p, q, indexes)

        alpha = r_norm / p_dot_q

        x, r, r_norm_new = y_pm_ax_numba(alpha, p, x0, q, r, indexes)

        if np.sqrt(r_norm_new) < atol:
            return x, iteration

        beta = r_norm_new / r_norm
        p = ax_p_y_numba(beta, p, r, indexes, p)
        r_norm = r_norm_new

    return x, iteration


def preconditioned_cg(A, b, x0, M, rtol=None, atol=1e-6, maxiter=100):
    """ Preconditioned Conjugate Gradient solver for A@x = b for x, where A is
    a sparse matrix in CSR format.

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
    M : object
        The preconditioner to use. Must have a method `matvec(r, z, indexes)`
        that applies the preconditioner to a vector r and stores the result in z.
    rtol : float
        Relative tolerance for convergence. The solver will stop when the
        residual norm is less than max(atol, rtol * ||b||).
        If None, only the absolute tolerance is used.
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
    precondtioner = M.matvec

    b_norm_2 = dot_numba(b, b, indexes)

    if b_norm_2 == 0:
        return x, 0
    
    if rtol is not None:
        atol = max(atol, rtol * np.sqrt(b_norm_2))

    
    r = np.empty_like(x)
    r = b_m_matvec_numba(indptr, indices, data, b, x, r, indexes)
    q = np.empty_like(r)
    z = np.empty_like(r)

    rho_prev, p = None, None

    for iteration in range(maxiter):

        r_norm_2 = dot_numba(r, r, indexes)
        if np.sqrt(r_norm_2) < atol:  # Are we done?
            return x, iteration
        
        z = precondtioner(r, z, indexes)
        rho_cur = dot_numba(r, z, indexes)

        if iteration > 0:
            beta = rho_cur / rho_prev
            p = ax_p_y_numba(beta, p, r, indexes, p)
        else:  # First spin
            p = np.empty_like(r)
            p[:] = z[:]

        q, p_dot_q = matvec_and_dot_numba(indptr, indices, data, p, q, indexes)
        alpha = rho_cur / p_dot_q
        rho_prev = rho_cur
        x, r, rho_cur = y_pm_ax_numba(alpha, p, x, q, r, indexes)

    return x, -1


@njit(parallel=True, fastmath=True, cache=True)
def matvec_p_ay_numba(indptr, indices, data, x, y, a, indexes, out):
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
def matvec_numba(indptr, indices, data, x, out, indexes):
    """
    Computes out = A @ x for a sparse matrix A in CSR format.
    """
    n = indexes.shape[0]
    for ii in prange(n):
        i = indexes[ii]
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
def matvec_and_dot_numba(indptr, indices, data, x, y, indexes):
    """Computes y = A @ x and returns the dot product x^T @ (A @ x).
    """
    n = indptr.shape[0] - 1
    s = 0.
    for i in prange(n):
        start, end = indptr[i], indptr[i+1]
        if start == end:
            continue
        y_i = 0.
        for j in range(start, end):
            jj = indices[j]
            jj = indexes[jj]
            y_i += data[j] * x.flat[jj]

        ii = indexes[i]
        y.flat[ii] = y_i
        s += y_i * x.flat[ii]
    return y, s


@njit(parallel=True, fastmath=True, cache=True)
def b_m_matvec_numba(indptr, indices, data, b, x, y, indexes):
    """
    Computes y = b - A @ x
    """
    n = indptr.shape[0] - 1
    for i in prange(n):
        start, end = indptr[i], indptr[i+1]
        if start == end:
            continue
        y_i = 0.
        for j in range(start, end):
            jj = indices[j]
            jj = indexes[jj]
            y_i += data[j] * x.flat[jj]

        ii = indexes[i]
        y.flat[ii] = b.flat[ii] - y_i
    return y


@njit(parallel=True, fastmath=True, cache=True)
def dot_numba(x, y, indexes):
    n = len(indexes)
    s = 0.
    for i in prange(n):
        ii = indexes[i]
        s += x.flat[ii] * y.flat[ii]
    return s


@njit(parallel=True, fastmath=True, cache=True)
def ax_m_y_numba(a, x, y, indexes, out):
    """Computes out = a * x - y in place.
    """
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        out.flat[ii] = a * x.flat[ii] - y.flat[ii]
    return out


@njit(parallel=True, fastmath=True, cache=True)
def ax_p_y_numba(a, x, y, indexes, out):
    """Computes out = a * x + y in place.
    """
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        out.flat[ii] = a * x.flat[ii] + y.flat[ii]
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
def copyto_numba(x, y, indexes):
    """
    Copies x to y in place for the given indexes.
    """
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        y.flat[ii] = x.flat[ii]
    return y
    