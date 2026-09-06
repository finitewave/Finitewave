"""Backend interface for linear-algebra operations used by time integration.

Backend modules implement these functions using their native array and sparse
matrix representations.
"""


def select_explicit_solver(x, active_indexes):
    """Select an explicit-step implementation for a solution layout.

    Parameters
    ----------
    x : array-like
        Solution vector defined either on the full domain or only at active
        cells.
    active_indexes : array-like
        Integer indexes of the active cells.

    Returns
    -------
    callable
        A backend function implementing :func:`explicit_step`.

    Raises
    ------
    ValueError
        If the solution and active-index layouts are incompatible.
    """
    raise NotImplementedError("select_explicit_solver must be implemented.")


def explicit_step(A_x, x, A_y, y, active_mask, out):
    """Compute one explicit update, ``out = A_x @ x + A_y @ y``.

    Parameters
    ----------
    A_x : backend sparse matrix
        Matrix applied to the solution vector.
    x : array-like
        Solution vector.
    A_y : backend sparse matrix
        Matrix applied to the reaction term.
    y : array-like
        Reaction-term vector.
    active_mask : array-like
        Boolean mask identifying active cells.
    out : array-like
        Output buffer for the updated solution.

    Returns
    -------
    array-like
        The updated solution.
    """
    raise NotImplementedError("explicit_step must be implemented.")


def cg(A, b, x0, *, atol=1e-8, maxiter=None):
    """Solve ``A @ x = b`` with the Conjugate Gradient method.

    Parameters
    ----------
    A : backend sparse matrix
        Symmetric positive-definite system matrix.
    b : array-like
        Right-hand-side vector.
    x0 : array-like
        Initial solution estimate.
    atol : float, optional
        Absolute convergence tolerance.
    maxiter : int, optional
        Maximum number of iterations.

    Returns
    -------
    x : array-like
        Approximate solution.
    num_iterations : int
        Number of iterations, or a backend-specific failure indicator.
    """
    raise NotImplementedError("cg must be implemented.")


def prepare_implicit_step(A_rhs, A_reaction, x_old, x_old_2, reaction_term, active_indexes):
    """Prepare the initial estimate and right-hand side for an implicit step.

    Computes ``x0 = 2 * x_old - x_old_2`` and
    ``b = A_rhs @ x_old + A_reaction @ reaction_term``. Returned vectors are
    restricted to active cells when required by the backend.

    Parameters
    ----------
    A_rhs : backend sparse matrix
        Matrix applied to the previous solution.
    A_reaction : backend sparse matrix
        Matrix applied to the reaction term.
    x_old : array-like
        Solution vector from the previous time step.
    x_old_2 : array-like
        Solution vector from two time steps ago.
    reaction_term : array-like
        Reaction term computed by the cardiac model.
    active_indexes : array-like
        Integer indexes of the active cells.

    Returns
    -------
    x0 : array-like
        Initial estimate for the implicit solver.
    b : array-like
        Right-hand-side vector for the implicit solver.
    """
    raise NotImplementedError("prepare_implicit_step must be implemented.")


def update_at_active_indexes(x, active_indexes, out):
    """Write an active-cell solution into a full-domain output vector.

    Parameters
    ----------
    x : array-like
        Updated values at active cells.
    active_indexes : array-like
        Integer indexes of the active cells.
    out : array-like
        Full-domain output vector.

    Returns
    -------
    array-like
        The updated full-domain solution.
    """
    raise NotImplementedError("update_at_active_indexes must be implemented.")
