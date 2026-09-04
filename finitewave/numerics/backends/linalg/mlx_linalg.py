import mlx.core as mx


def matvec(A, x):
    return matvec_ellpack(A, x)


def select_explicit_solver(x, active_indexes):
    if x.size == active_indexes.size:
        return explicit_step
    
    if x.size > active_indexes.size:
        return explicit_step_indexed

    raise ValueError("Invalid combination of x and active_indexes sizes.")


@mx.compile
def explicit_step(A_x, x, A_y, y, active_indexes, out):
    return matvec(A_x, x) + matvec(A_y, y)


@mx.compile
def explicit_step_indexed(A_x, x, A_y, y, active_mask, out):
    out = mx.where(active_mask, matvec(A_x, x) + matvec(A_y, y), out)
    return out


@mx.compile
def prepare_implicit_step(A_rhs, A_ion, x_old, x_old_2, i_ion, active_indexes):
    x0 = 2. * x_old[active_indexes] - x_old_2[active_indexes]
    b = matvec(A_rhs, x_old) + matvec(A_ion, i_ion)
    return x0, b


@mx.compile
def update_at_active_indexes(x, active_indexes, out):
    out[active_indexes] = x
    return out


def cg(A, b, x0, *, atol=1e-8, maxiter=1):
    """
    Conjugate Gradient solver for Ax = b where A is represented in ELLPACK format.
    
    Parameters
    ----------
    A : tuple
        A tuple containing (indices, data) representing the matrix in ELLPACK format.
    b : 1D array of float
        Right-hand side vector.
    x0 : 1D array of float, optional
        Initial guess for the solution. If None, a zero vector is used.
    indexes : 1D array of int, optional
        Array of indexes where the solution is defined.
        If None, the solution is defined for all entries.
    atol : float, optional
        Absolute tolerance for convergence.
    maxiter : int, optional
        Maximum number of iterations.
        
    Returns
    -------
    x : 1D array of float
        Approximate solution vector.
    int
        Number of iterations performed, or -1 if not converged within maxiter.
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
    """
    Prepares the initial residual and direction for the Conjugate Gradient solver.

    Parameters
    ----------
    A : tuple
        A tuple containing (indices, data) representing the matrix in ELLPACK format.
    b : 1D array of float
        Right-hand side vector.
    x0 : 1D array of float
        Initial guess for the solution.

    Returns
    -------
    r : 1D array of float
        Initial residual vector.
    p : 1D array of float
        Initial direction vector.
    r_norm : float
        Norm of the initial residual.
    """
    r = b - matvec(A, x0)
    p = r
    r_norm = mx.sum(r * r)
    return x0, r, p, r_norm

@mx.compile
def cg_n_steps(A, x, r, p, r_norm, n_steps):
    for _ in range(n_steps):
        x, r, p, r_norm = cg_step(A, x, r, p, r_norm)

    return x, r, p, r_norm


@mx.compile
def cg_step(A, x, r, p, r_norm):
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
    """
    Performs A @ x where A is represented in COO format.

    Parameters
    ----------
    A : tuple
        A tuple containing (row, col, data) representing the matrix in COO format.
    x : 1D array of float
        Input vector to be multiplied by A.

    Returns
    -------
    np.ndarray (len(row),)
        Result of the matrix-vector product.
    """
    row, col, data = A
    out = mx.zeros(x.shape, dtype=x.dtype)
    out = out.at[row].add(data * x[col])
    return out


@mx.compile
def matvec_ellpack(A, x):
    """
    Performs A @ x where A is represented in ELLPACK format.

    Parameters
    ----------
    A : tuple
        A tuple containing (indices, data) representing the matrix in ELLPACK format.
    x : 1D array of float
        Input vector to be multiplied by A.

    Returns
    -------
    np.ndarray (len(indexes),)
        Result of the matrix-vector product.
    """
    indices, data = A
    out = mx.sum(data * x[indices], axis=1)
    return out
