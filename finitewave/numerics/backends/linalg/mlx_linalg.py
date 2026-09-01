import mlx.core as mx


def select_explicit_solver(x, active_indexes):
    if x.size == active_indexes.size:
        return explicit_step
    
    if x.size > active_indexes.size:
        return explicit_step_indexed

    raise ValueError("Invalid combination of x and active_indexes sizes.")


@mx.compile
def explicit_step(A_x, x, A_y, y, active_indexes, out):
    return matvec_ellpack(A_x, x) + matvec_ellpack(A_y, y)


@mx.compile
def explicit_step_indexed(A_x, x, A_y, y, active_indexes, out):
    out[active_indexes] = matvec_ellpack(A_x, x) + matvec_ellpack(A_y, y)
    return out


def implicit_step(method, A_lhs, A_rhs, A_ion, x_old, x_old_2, i_ion, active_indexes, out, atol=1e-8, maxiter=100):
    x0, b = prepare_implicit_step(A_rhs, A_ion, x_old, x_old_2, i_ion, active_indexes)
    x_new, _ = method(A_lhs, b, x0, atol=atol, maxiter=maxiter)
    out = update_active_indexes(x_new, active_indexes, out)
    
    return out

@mx.compile
def prepare_implicit_step(A_rhs, A_ion, x_old, x_old_2, i_ion, active_indexes):
    x0 = 2. * x_old[active_indexes] - x_old_2[active_indexes]
    b = matvec_ellpack(A_rhs, x_old) + matvec_ellpack(A_ion, i_ion)
    return x0, b

@mx.compile
def update_active_indexes(x, active_indexes, out):
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
    indices, data = A
    x = x0
    r = b - matvec_ellpack(A, x)
    p = r
    r_norm = mx.sum(r * r)

    for i in range(maxiter):

        x, r, p, r_norm = cg_step(A, x, r, p, r_norm)
        
        # Convergence check: Only sync with CPU every 10-20 iterations
        if i % 10 == 0:
            mx.eval(r_norm)
            mx.synchronize()
            if mx.sqrt(r_norm) < atol:
                break
        
    return x, i


@mx.compile
def cg_step(A, x, r, p, r_norm):
    # 1. SpMV: Ap = A @ p
    Ap = matvec_ellpack(A, p)

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
