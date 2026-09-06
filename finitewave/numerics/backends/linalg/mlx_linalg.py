"""MLX implementations of linear-algebra operations used by Finitewave."""

import mlx.core as mx


def matvec(A, x):
    """Compute ``A @ x`` for an ELLPACK matrix."""
    return matvec_ellpack(A, x)


def select_explicit_solver(x, active_indexes):
    """Select an explicit-step function for the MLX solution layout.

    Parameters
    ----------
    x : mx.array
        Solution array.
    active_indexes : mx.array
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


@mx.compile
def explicit_step(A_x, x, A_y, y, active_indexes, out):
    """Compute ``A_x @ x + A_y @ y`` over the full domain.

    ``active_indexes`` and ``out`` are retained for the common backend
    interface but are not used by this full-domain implementation.

    Parameters
    ----------
    A_x, A_y : tuple
        ELLPACK arrays ``(indices, data)``.
    x, y : mx.array
        Solution and reaction-term vectors, respectively.
    active_indexes : mx.array
        Active-cell selector retained for interface compatibility.
    out : mx.array
        Output buffer retained for interface compatibility.

    Returns
    -------
    mx.array
        Result of the explicit update.
    """
    return matvec(A_x, x) + matvec(A_y, y)


@mx.compile
def explicit_step_indexed(A_x, x, A_y, y, active_mask, out):
    """Compute an explicit update while retaining inactive values.

    Parameters
    ----------
    A_x, A_y : tuple
        ELLPACK arrays ``(indices, data)``.
    x, y : mx.array
        Solution and reaction-term vectors, respectively.
    active_mask : mx.array
        Boolean mask identifying active cells.
    out : mx.array
        Values to retain at inactive cells.

    Returns
    -------
    mx.array
        Updated solution with inactive values preserved.
    """
    out = mx.where(active_mask, matvec(A_x, x) + matvec(A_y, y), out)
    return out


@mx.compile
def prepare_implicit_step(
        A_rhs, A_reaction, x_old, x_old_2, reaction_term, active_indexes):
    """Prepare the initial estimate and right-hand side for an implicit step.

    Parameters
    ----------
    A_rhs, A_reaction : tuple
        ELLPACK arrays for the solution and reaction matrices.
    x_old : mx.array
        Solution vector from the previous time step.
    x_old_2 : mx.array
        Solution vector from two time steps ago.
    reaction_term : mx.array
        Reaction term computed by the cardiac model.
    active_indexes : mx.array
        Integer indexes of active cells.

    Returns
    -------
    x0 : mx.array
        Extrapolated initial estimate at active cells.
    b : mx.array
        Linear-system right-hand side.
    """
    x0 = 2. * x_old[active_indexes] - x_old_2[active_indexes]
    b = matvec(A_rhs, x_old) + matvec(A_reaction, reaction_term)
    return x0, b


@mx.compile
def update_at_active_indexes(x, active_indexes, out):
    """Write active-cell values into a full-domain output array.

    Parameters
    ----------
    x : mx.array
        Updated values at active cells.
    active_indexes : mx.array
        Integer indexes of active cells.
    out : mx.array
        Full-domain output array.

    Returns
    -------
    mx.array
        ``out`` with active entries updated.
    """
    out[active_indexes] = x
    return out


def cg(A, b, x0, *, atol=1e-8, maxiter=1):
    """Solve ``A @ x = b`` using the Conjugate Gradient method.
    
    Parameters
    ----------
    A : tuple
        ELLPACK arrays ``(indices, data)`` for a symmetric positive-definite
        matrix.
    b : mx.array
        Right-hand side vector.
    x0 : mx.array
        Initial solution estimate.
    atol : float, optional
        Absolute tolerance for convergence.
    maxiter : int, optional
        Maximum number of iterations.
        
    Returns
    -------
    x : mx.array
        Approximate solution vector.
    iteration : int
        Zero-based index of the final iteration.
    """
    # Initial residual and direction
    # r = b - A@x
    x, r, p, r_norm = prepare_cg(A, b, x0)

    for i in range(maxiter):

        x, r, p, r_norm = cg_step(A, x, r, p, r_norm)

        if i % 10 == 0:
            mx.eval(r_norm)
            mx.synchronize()
            if mx.sqrt(r_norm) < atol:
                break
        
    return x, i

@mx.compile
def prepare_cg(A, b, x0):
    """Initialize the state of the Conjugate Gradient iteration.

    Parameters
    ----------
    A : tuple
        ELLPACK arrays ``(indices, data)`` for the system matrix.
    b : mx.array
        Right-hand side vector.
    x0 : mx.array
        Initial solution estimate.

    Returns
    -------
    x : mx.array
        Initial solution estimate.
    r : mx.array
        Initial residual vector.
    p : mx.array
        Initial direction vector.
    r_norm : mx.array
        Squared Euclidean norm of the initial residual.
    """
    r = b - matvec(A, x0)
    p = r
    r_norm = mx.sum(r * r)
    return x0, r, p, r_norm

@mx.compile
def cg_n_steps(A, x, r, p, r_norm, n_steps):
    """Run a fixed number of Conjugate Gradient iterations.

    Parameters
    ----------
    A : tuple
        ELLPACK arrays ``(indices, data)`` for the system matrix.
    x, r, p : mx.array
        Current solution, residual, and search-direction vectors.
    r_norm : mx.array
        Squared Euclidean norm of ``r``.
    n_steps : int
        Number of iterations to execute.

    Returns
    -------
    x : mx.array
        Updated solution estimate.
    r : mx.array
        Updated residual.
    p : mx.array
        Updated search direction.
    r_norm : mx.array
        Updated squared residual norm.
    """
    for _ in range(n_steps):
        x, r, p, r_norm = cg_step(A, x, r, p, r_norm)

    return x, r, p, r_norm


@mx.compile
def cg_step(A, x, r, p, r_norm):
    """Perform one Conjugate Gradient iteration.

    Parameters
    ----------
    A : tuple
        ELLPACK arrays ``(indices, data)`` for the system matrix.
    x, r, p : mx.array
        Current solution, residual, and search-direction vectors.
    r_norm : mx.array
        Squared Euclidean norm of ``r``.

    Returns
    -------
    x : mx.array
        Updated solution estimate.
    r : mx.array
        Updated residual.
    p : mx.array
        Updated search direction.
    r_norm : mx.array
        Updated squared residual norm.
    """
    # 1. SpMV: Ap = A @ p
    Ap = matvec(A, p)

    # 2. Alpha: step size
    alpha = r_norm / mx.sum(p * Ap)
    
    # 3. Update solution and residual
    x = x + alpha * p
    r = r - alpha * Ap
    
    # 4. Beta: direction update
    r_norm_new = mx.sum(r * r)
    p = r + (r_norm_new / r_norm) * p
    
    return x, r, p, r_norm_new


@mx.compile
def matvec_coo(A, x):
    """Compute ``A @ x`` for a COO matrix.

    Parameters
    ----------
    A : tuple
        COO arrays ``(row, col, data)``.
    x : mx.array
        Input vector.

    Returns
    -------
    mx.array
        Matrix-vector product with the same shape and dtype as ``x``.
    """
    row, col, data = A
    out = mx.zeros(x.shape, dtype=x.dtype)
    out = out.at[row].add(data * x[col])
    return out


@mx.compile
def matvec_ellpack(A, x):
    """Compute ``A @ x`` for an ELLPACK matrix.

    Parameters
    ----------
    A : tuple
        ELLPACK arrays ``(indices, data)``. Each row of ``indices`` contains
        column indexes corresponding to the values in ``data``.
    x : mx.array
        Input vector.

    Returns
    -------
    mx.array
        Matrix-vector product with one value per matrix row.
    """
    indices, data = A
    out = mx.sum(data * x[indices], axis=1)
    return out
