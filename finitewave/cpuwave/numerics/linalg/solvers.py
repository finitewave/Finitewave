import numpy as np
from numba import njit, prange
from .numba_linalg import (
    dot_numba,
    b_m_matvec_numba,
    ay_p_x_numba,
    matvec_and_dot_numba,
    y_pm_ax_numba)


def forward_euler(A, u, rhs, mass_inv, u_new, indexes, dt):
    """Performs the Forward Euler step for the diffusion equation.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The stiffness matrix representing the diffusion operator.
    u : np.ndarray
        Current solution vector.
    rhs : np.ndarray
        Right-hand side vector (e.g., source term).
    mass_inv : np.ndarray
        Inverse of the lumped mass matrix diagonal.
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
    return forward_euler_numba(indptr, indices, data, u, rhs, mass_inv, u_new, indexes, dt)


@njit(parallel=True, fastmath=True, cache=True)
def forward_euler_numba(indptr, indices, data, u, rhs, mass_inv, u_new, indexes, dt):
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
    mass_inv : np.ndarray
        Inverse of the lumped mass matrix diagonal.
    u_new : np.ndarray
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
        i_tr = 0.0
        for j in range(start, end):
            jj = indices[j]
            jj = indexes[jj]
            i_tr += data[j] * u.flat[jj]

        ii = indexes[i]
        u_new.flat[ii] = (u.flat[ii] -
                          dt * mass_inv.flat[i] * i_tr +
                          dt * rhs.flat[ii])

    return u_new


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
