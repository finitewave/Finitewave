import numpy as np
from numba import njit, prange


def forward_euler(A, u, rhs, u_new, indexes, dt):
    """Performs the Forward Euler step for the diffusion equation.

    For each time step:
        1. Compute the diffusion term diff = A @ u.
        2. Update the solution using u_new = u - diff + dt * rhs

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The system matrix representing the diffusion operator.
    u : np.ndarray
        Current solution vector.
    rhs : np.ndarray
        Right-hand side vector (e.g., source term).
    u_new : np.ndarray
        Output solution vector to store the updated solution.
    indexes : np.ndarray
        Array of indexes where the solution is defined.
    dt : float
        Time step size.

    Returns
    -------
    np.ndarray
        Updated solution vector after the Forward Euler step.
    """
    indptr, indices, data = A.indptr, A.indices, A.data
    return forward_euler_numba(indptr, indices, data, u, rhs, u_new, indexes, dt)


@njit(parallel=True, fastmath=True, cache=True)
def forward_euler_numba(indptr, indices, data, x, f, x_new, indexes, dt):
    """
    Performs the Forward Euler step:
    x_new = x + A @ x + dt * f

    Parameters
    ----------
    indptr : np.ndarray
        CSR index pointer array.
    indices : np.ndarray
        CSR indices array.
    data : np.ndarray
        CSR data array.
    x : np.ndarray
        Current solution vector.
    f : np.ndarray
        Source term vector.
    x_new : np.ndarray
        Output solution vector.
    indexes : np.ndarray
        Array of indexes where the solution is defined.

    Returns
    -------
    x_new : np.ndarray
        Updated solution vector after the Forward Euler step.
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
        x_new.flat[ii] = x.flat[ii] - a_x + dt * f.flat[ii]

    return x_new


def cg_numba(A, b, x, indexes, rtol=None, atol=1e-6, maxiter=100):
    """ Conjugate Gradient solver for Ax = b for x, where A is a sparse
    matrix given in CSR format.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The matrix A in Ax = b.
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
    indptr, indices, data = A.indptr, A.indices, A.data
    b_norm_2 = dot_numba(b, b, indexes)

    if b_norm_2 == 0:
        return x, 0
    
    if rtol is not None:
        atol = max(atol, rtol * np.sqrt(b_norm_2))

    r = np.empty_like(x)
    r = b_m_matvec_numba(indptr, indices, data, b, x, r, indexes)
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


def preconditioned_cg_numba(A, b, x, indexes, M, rtol=None, atol=1e-6, maxiter=100):
    """ Preconditioned Conjugate Gradient solver for A@x = b for x, where A is
    a sparse matrix in CSR format.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The matrix A in Ax = b.
    b : 1D array of float
        Right-hand side vector.
    x : 1D array of float
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
    indptr, indices, data = A.indptr, A.indices, A.data
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
            p = ay_p_x_numba(beta, z, p, indexes)
        else:  # First spin
            p = np.empty_like(r)
            p[:] = z[:]

        q, p_dot_q = matvec_and_dot_numba(indptr, indices, data, p, q, indexes)
        alpha = rho_cur / p_dot_q
        x, r = y_pm_ax_numba(alpha, p, x, q, r, indexes)
        rho_prev = rho_cur

    return x, -1

@njit(parallel=True, fastmath=True, cache=True)
def matvec_numba(indptr, indices, data, x, y, indexes):
    """
    Computes y = A @ x for a sparse matrix A in CSR format.
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
        y.flat[ii] = y_i
    return y


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
def ay_p_x_numba(a, x, y, indexes):
    """Computes y = a * y + x in place.
    """
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        y.flat[ii] = a * y.flat[ii] + x.flat[ii]
    return y


@njit(parallel=True, fastmath=True, cache=True)
def ax_p_y_numba(a, x, y, indexes):
    """Computes y = a * x + y in place.
    """
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        y.flat[ii] = a * x.flat[ii] + y.flat[ii]
    return y


@njit(parallel=True, fastmath=True, cache=True)
def y_pm_ax_numba(a, xp, yp, xm, ym, indexes):
    """
    Computes yp = yp + a * xp and ym = ym - a * xm in place.
    """
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        yp.flat[ii] += a * xp.flat[ii]
        ym.flat[ii] -= a * xm.flat[ii]
    return yp, ym


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
    