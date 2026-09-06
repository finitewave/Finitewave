"""JAX implementations of linear-algebra operations used by Finitewave."""

from functools import partial
import jax
import jax.numpy as jnp


def select_explicit_solver(x, active_indexes):
    """Select an explicit-step function for the JAX solution layout.

    Parameters
    ----------
    x : jax.Array
        Solution array.
    active_indexes : jax.Array
        Integer indexes of active cells.

    Returns
    -------
    callable
        Either the full-domain or masked explicit-step function.

    Raises
    ------
    ValueError
        If ``x`` has fewer entries than ``active_indexes``.
    """
    if x.size == active_indexes.size:
        return explicit_step
    
    if x.size > active_indexes.size:
        return explicit_step_indexed

    raise ValueError("Invalid combination of x and active_indexes sizes.")


@jax.jit
def explicit_step(A_x, x, A_y, y, active_indexes, out):
    """Compute ``A_x @ x + A_y @ y`` over the full domain.

    Parameters
    ----------
    A_x : tuple
        ELLPACK arrays ``(indices, data)`` for the solution matrix.
    x : jax.Array
        Solution vector multiplied by ``A_x``.
    A_y : tuple
        ELLPACK arrays ``(indices, data)`` for the reaction matrix.
    y : jax.Array
        Reaction-term vector multiplied by ``A_y``.
    active_indexes : jax.Array
        Active-cell selector retained for the common backend interface.
        This full-domain implementation does not use it.
    out : jax.Array
        Output buffer retained for the common backend interface. JAX returns a
        new array instead of modifying this buffer.

    Returns
    -------
    jax.Array
        Result of the explicit update.
    """
    return matvec_ellpack(A_x, x) + matvec_ellpack(A_y, y)


@jax.jit
def explicit_step_indexed(A_x, x, A_y, y, active_indexes, out):
    """Compute an explicit update and retain inactive values from ``out``.

    Parameters
    ----------
    A_x : tuple
        ELLPACK arrays ``(indices, data)`` for the solution matrix.
    x : jax.Array
        Solution vector multiplied by ``A_x``.
    A_y : tuple
        ELLPACK arrays ``(indices, data)`` for the reaction matrix.
    y : jax.Array
        Reaction-term vector multiplied by ``A_y``.
    active_indexes : jax.Array
        Boolean mask identifying active cells.
    out : jax.Array
        Values to retain at inactive cells.

    Returns
    -------
    jax.Array
        Updated solution with inactive values preserved.
    """
    out = jnp.where(active_indexes, matvec_ellpack(A_x, x) + matvec_ellpack(A_y, y), out)
    return out


@jax.jit
def cg(A, b, x0=None, *, tol=0.0, atol=1e-8, maxiter=None, M=None):
    """Solve ``A @ x = b`` using JAX's Conjugate Gradient implementation.

    Parameters
    ----------
    A : tuple
        ELLPACK arrays ``(indices, data)`` for a symmetric positive-definite
        matrix.
    b : jax.Array
        Right-hand-side vector.
    x0 : jax.Array, optional
        Initial solution estimate.
    tol : float, optional
        Relative convergence tolerance.
    atol : float, optional
        Absolute convergence tolerance.
    maxiter : int, optional
        Maximum number of iterations.
    M : callable, optional
        Preconditioner accepted by :func:`jax.scipy.sparse.linalg.cg`.

    Returns
    -------
    x : jax.Array
        Approximate solution.
    info : int
        Compatibility status value. This wrapper currently returns 0.
    """
    matvec = lambda x: matvec_ellpack(A, x)
    x, info = jax.scipy.sparse.linalg.cg(matvec, b, x0=x0, tol=tol, atol=atol,
                                         maxiter=maxiter, M=M)
    return x, 0


@jax.jit
def prepare_implicit_step(A_rhs, A_reaction, x_old, x_old_2, reaction_term, active_indexes):
    """Prepare the initial estimate and right-hand side for an implicit step.

    Parameters
    ----------
    A_rhs : tuple
        ELLPACK arrays for the matrix applied to ``x_old``. Column indexes use
        global domain indexing.
    A_reaction : tuple
        ELLPACK arrays for the matrix applied to ``reaction_term``. Column
        indexes use global domain indexing.
    x_old : jax.Array
        Solution vector from the previous time step.
    x_old_2 : jax.Array
        Solution vector from two time steps ago.
    reaction_term : jax.Array
        Reaction term computed by the cardiac model.
    active_indexes : jax.Array
        Integer indexes of active cells.

    Returns
    -------
    x0 : jax.Array
        Extrapolated initial estimate at active cells.
    b : jax.Array
        Linear-system right-hand side.
    """
    x0 = 2. * x_old[active_indexes] - x_old_2[active_indexes]
    b = (matvec_ellpack(A_rhs, x_old) +
         matvec_ellpack(A_reaction, reaction_term))
    return x0, b


@jax.jit
def update_at_active_indexes(x, active_indexes, out):
    """Write active-cell values into a full-domain output array.

    Parameters
    ----------
    x : jax.Array
        Updated values at active cells.
    active_indexes : jax.Array
        Integer indexes of active cells.
    out : jax.Array
        Full-domain output array.

    Returns
    -------
    jax.Array
        Copy of ``out`` with active entries updated.
    """
    out = out.at[active_indexes].set(x)
    return out


@jax.jit
def matvec_ellpack(A, x):
    """Compute ``A @ x`` for an ELLPACK matrix.

    Parameters
    ----------
    A : tuple
        ELLPACK arrays ``(indices, data)``. Each row of ``indices`` contains
        column indexes corresponding to the values in ``data``.
    x : jax.Array
        Input vector.

    Returns
    -------
    jax.Array
        Matrix-vector product with one value per matrix row.
    """
    indices, data = A
    out = jnp.sum(data * x[indices], axis=1)
    return out
