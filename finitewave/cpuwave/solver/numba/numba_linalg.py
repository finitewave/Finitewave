import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def forward_euler(indptr, indices, data, u, rhs, mass_inv, u_new, indexes, dt):
    """
    Performs the Forward Euler step:
    y = u + M^{-1} * (A * u + rhs)

    Parameters
    ----------
    indptr : np.ndarray
        CSR index pointer array.
    indices : np.ndarray
        CSR indices array.
    data : np.ndarray
        CSR data array.
    u : np.ndarray
        Current solution vector.
    rhs : np.ndarray
        Right-hand side vector.
    mass : np.ndarray
        Inverse of the mass matrix diagonal.
    y : np.ndarray
        Output solution vector.
    indexes : np.ndarray
        Array of indexes where the solution is defined.

    Returns
    -------
    y : np.ndarray
        Updated solution vector after the Forward Euler step.
    """
    n_rows = indptr.size - 1

    for i in prange(n_rows):
        start, end = indptr[i], indptr[i+1]
        if start == end:
            continue
        tr_curr = 0.0
        for j in range(start, end):
            jj = indices[j]
            jj = indexes[jj]
            tr_curr += data[j] * u.flat[jj]

        ii = indexes[i]
        u_new.flat[ii] = u.flat[ii] + dt * (mass_inv.flat[ii] * (-tr_curr) +
                                            rhs.flat[ii])

    return u_new


@njit(parallel=True, fastmath=True, cache=True)
def matvec_numba(indptr, indices, data, x, y, indexes):
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
        y.flat[ii] = y_i
    return y


@njit(parallel=True, fastmath=True, cache=True)
def matvec_and_dot_numba(indptr, indices, data, x, y, indexes):
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
def dot_numba(x, y, indexes):
    n = len(indexes)
    s = 0.
    for i in prange(n):
        ii = indexes[i]
        s += x.flat[ii] * y.flat[ii]
    return s


@njit(parallel=True, fastmath=True, cache=True)
def ay_p_x_numba(a, x, y, indexes):
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        y.flat[ii] = a * y.flat[ii] + x.flat[ii]
    return y


@njit(parallel=True, fastmath=True, cache=True)
def y_pm_ax_numba(a, xp, yp, xm, ym, indexes):
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        yp.flat[ii] += a * xp.flat[ii]
        ym.flat[ii] -= a * xm.flat[ii]
    return yp, ym


def cg_numba(indptr, indices, data, b, x, indexes, atol=1e-6, maxiter=100):
    """ Conjugate Gradient solver for Ax = b for x, where A is given by
    (indptr, indices, data) in CSR format.

    Parameters
    ----------
    indptr : 1D array of int
        CSR index pointer array.
    indices : 1D array of int
        CSR indices array.
    data : 1D array of float
        CSR data array.
    b : 1D array of float
        Right-hand side vector.
    x : 1D array of float
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

    if np.sqrt(dot_numba(b, b, indexes)) == 0:
        return x, 0

    b0 = np.empty_like(x)
    b0, _ = matvec_and_dot_numba(indptr, indices, data, x, b0, indexes)
    r = b - b0
    q = np.empty_like(r)

    # Dummy value to initialize var, silences warnings
    rho_prev, p = None, None

    for iteration in range(maxiter):
        rho_cur = dot_numba(r, r, indexes)
        if np.sqrt(rho_cur) < atol:  # Are we done?
            return x, iteration

        if iteration > 0:
            beta = rho_cur / rho_prev
            p = ay_p_x_numba(beta, r, p, indexes)
        else:  # First spin
            p = np.empty_like(r)
            p[:] = r[:]

        q, p_dot_q = matvec_and_dot_numba(indptr, indices, data, p, q, indexes)
        alpha = rho_cur / p_dot_q
        x, r = y_pm_ax_numba(alpha, p, x, q, r, indexes)
        rho_prev = rho_cur

    return x, -1
