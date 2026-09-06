"""Numba implementations of linear-algebra operations used by Finitewave."""

import numpy as np
from numba import njit, prange


def select_explicit_solver(x, active_indexes):
    """Select the Numba explicit-step function for a solution layout.

    Parameters
    ----------
    x : np.ndarray
        Solution array.
    active_indexes : np.ndarray
        Flat integer indexes of active cells.

    Returns
    -------
    callable
        The Numba explicit-step function.

    Raises
    ------
    ValueError
        If ``x`` has fewer entries than ``active_indexes``.
    """
    if x.size >= active_indexes.size:
        return explicit_step
    
    raise ValueError("Invalid combination of x and active_indexes sizes.")


def explicit_step(A_x, x, A_y, y, active_mask, out):
    """Compute ``out = A_x @ x + A_y @ y`` for one explicit step.

    Parameters
    ----------
    A_x : tuple
        CSR arrays ``(indptr, indices, data)`` for the solution matrix.
    x : np.ndarray
        Solution vector multiplied by ``A_x``.
    A_y : tuple
        CSR arrays ``(indptr, indices, data)`` for the reaction matrix.
    y : np.ndarray
        Reaction-term vector multiplied by ``A_y``.
    active_mask : np.ndarray
        Boolean active-cell mask retained for the common backend interface.
        This implementation does not use it directly.
    out : np.ndarray
        Output buffer for the updated solution.

    Returns
    -------
    np.ndarray
        The updated output buffer.
    """
    out = matvec_numba(A_x[0], A_x[1], A_x[2], x, out)
    return matvec_p_ay_numba(A_y[0], A_y[1], A_y[2], y, 1.0, out, out)


def prepare_implicit_step(
        A_rhs, A_reaction, x_old, x_old_2, reaction_term, active_indexes):
    """Prepare the initial estimate and right-hand side for an implicit step.

    Parameters
    ----------
    A_rhs : tuple
        CSR arrays for the matrix applied to ``x_old``.
    A_reaction : tuple
        CSR arrays for the matrix applied to ``reaction_term``.
    x_old : np.ndarray
        Solution vector from the previous time step.
    x_old_2 : np.ndarray
        Solution vector from two time steps ago.
    reaction_term : np.ndarray
        Reaction term computed by the cardiac model.
    active_indexes : np.ndarray
        Flat integer indexes of active cells.

    Returns
    -------
    x0 : np.ndarray
        Extrapolated initial estimate, restricted to active cells if needed.
    b : np.ndarray
        Linear-system right-hand side, restricted to active cells if needed.
    """
    x0 = np.empty(x_old.size, dtype=x_old.dtype)
    x0 = ax_p_by_numba(2.0, x_old, -1.0, x_old_2, x0)

    b = np.empty(x_old.size, dtype=x_old.dtype)
    b = matvec_numba(A_reaction[0], A_reaction[1], A_reaction[2], reaction_term, b)
    b = matvec_p_ay_numba(A_rhs[0], A_rhs[1], A_rhs[2], x_old, 1.0, b, b)

    if active_indexes.size != x_old.size:
        x0 = select_values_numba(x0, active_indexes)
        b = select_values_numba(b, active_indexes)

    return x0, b

def update_at_active_indexes(x, active_indexes, out):
    """Write active-cell values into a full-domain output array.

    Parameters
    ----------
    x : np.ndarray
        Updated active-cell values, or a full-domain solution.
    active_indexes : np.ndarray
        Flat integer indexes of active cells.
    out : np.ndarray
        Full-domain output buffer.

    Returns
    -------
    np.ndarray
        ``out`` with active entries updated, or ``x`` when already full-sized.
    """
    if active_indexes.size != out.size:
        out = set_values_numba(x, active_indexes, out)
    else:
        out = x
    return out


def cg(A, b, x0, M=None, atol=1e-6, maxiter=100):
    """Solve ``A @ x = b`` using the Conjugate Gradient method.

    Parameters
    ----------
    A : tuple
        CSR arrays ``(indptr, indices, data)`` for a symmetric
        positive-definite matrix.
    b : np.ndarray
        Right-hand side vector.
    x0 : np.ndarray
        Initial solution estimate. It may be modified in place.
    M : optional
        Reserved for a preconditioner; currently unused.
    atol : float, optional
        Absolute tolerance for the stopping criterion.
    maxiter : int, optional
        Maximum number of iterations.

    Returns
    -------
    x : np.ndarray
        Approximate solution vector.
    iteration : int
        Zero-based index of the final iteration. Returns 0 when the initial
        estimate already satisfies the tolerance.
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
    """Compute ``out = A @ x + a * y`` for a CSR matrix.

    ``indices`` must use global indexing into ``x``.

    Parameters
    ----------
    indptr, indices, data : np.ndarray
        CSR representation of ``A``.
    x, y : np.ndarray
        Input arrays.
    a : float
        Scalar multiplier for ``y``.
    out : np.ndarray
        Output buffer.

    Returns
    -------
    np.ndarray
        The updated output buffer.
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
    """Compute ``out = A @ x`` for a CSR matrix.

    Parameters
    ----------
    indptr, indices, data : np.ndarray
        CSR representation of ``A``.
    x : np.ndarray
        Input array.
    out : np.ndarray
        Output buffer.

    Returns
    -------
    np.ndarray
        The updated output buffer.
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
    """Return the dot product of flattened arrays ``x`` and ``y``."""
    return np.dot(x, y)


@njit(parallel=True, fastmath=True, cache=True)
def ax_p_by_numba(a, x, b, y, out):
    """Compute ``out = a * x + b * y`` elementwise.

    Parameters
    ----------
    a, b : float
        Scalar multipliers.
    x, y : np.ndarray
        Input arrays with equal sizes.
    out : np.ndarray
        Output buffer.

    Returns
    -------
    np.ndarray
        The updated output buffer.
    """
    n = x.size
    for i in prange(n):
        out.flat[i] = a * x.flat[i] + b * y.flat[i]
    return out


@njit(parallel=True, fastmath=True, cache=True)
def copyto_numba(x, out):
    """Copy all flattened values from ``x`` into ``out``.

    Parameters
    ----------
    x : np.ndarray
        Source array.
    out : np.ndarray
        Destination array.

    Returns
    -------
    np.ndarray
        The updated destination array.
    """
    n = x.size
    for i in prange(n):
        out.flat[i] = x.flat[i]
    return out

@njit(parallel=True, fastmath=True, cache=True)
def select_values_numba(arr, inds):
    """Select flattened values from an array.

    Parameters
    ----------
    arr : np.ndarray
        Source array.
    inds : np.ndarray
        Flat integer indexes to select.

    Returns
    -------
    np.ndarray
        Values from ``arr`` at ``inds``.
    """
    out = np.empty(inds.shape, dtype=arr.dtype)
    for i in prange(inds.size):
        out[i] = arr[inds[i]]
    return out


@njit(parallel=True, fastmath=True, cache=True)
def set_values_numba(values, inds, arr):
    """Set flattened array values at specified indexes.

    Parameters
    ----------
    values : np.ndarray
        Values to assign.
    inds : np.ndarray
        Flat integer indexes at which to assign values.
    arr : np.ndarray
        Array to update in place.

    Returns
    -------
    np.ndarray
        The updated array.
    """
    for i in prange(inds.size):
        arr[inds[i]] = values[i]
    return arr
