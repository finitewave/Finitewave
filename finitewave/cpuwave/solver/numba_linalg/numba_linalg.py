import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def matvec_p_ay_numba(indptr, indices, data, x, y, a, indexes, out):
    """
    Performs the Forward Euler step:
    x_new = A @ x + a * y

    Parameters
    ----------
    indptr : np.ndarray
        CSR index pointer array.
    indices : np.ndarray
        CSR indices array.
    data : np.ndarray
        CSR data array.
    x : np.ndarray
        Input vector to be multiplied by A.
    y : np.ndarray
        Input vector to be scaled by a and added to A @ x.
    a : float
        Scalar multiplier for y.
    indexes : np.ndarray
        Array of indexes where the solution is defined.
    out : np.ndarray
        Output solution vector.

    Returns
    -------
    out : np.ndarray
        Vector with the result of A @ x + a * y for the given indexes.
    """
    n_rows = indptr.size - 1

    for i in prange(n_rows):
        start, end = indptr[i], indptr[i+1]
        if start == end:
            continue
        a_x = 0.0
        for j in range(start, end):
            jj = indices[j]
            jj = indexes[jj]
            a_x += data[j] * x.flat[jj]

        ii = indexes[i]
        out.flat[ii] = a_x + a * y.flat[ii]

    return out


@njit(parallel=True, fastmath=True, cache=True)
def matvec_numba(indptr, indices, data, x, out, indexes):
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
            jj = indexes[jj]
            out_i += data[j] * x.flat[jj]

        ii = indexes[i]
        out.flat[ii] = out_i
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
def ay_p_x_numba(a, x, y, indexes, out):
    """Computes out = a * y + x in place.
    """
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        out.flat[ii] = a * y.flat[ii] + x.flat[ii]
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
    