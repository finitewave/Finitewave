import mlx.core as mx


@mx.compile
def forward_euler_mlx(indices, data, u, u_old, rhs, indexes, dt):
    """
    Performs the Forward Euler step:
    u_new = A @ u_old + dt * rhs

    Parameters
    ----------
    data : 2D array of float
        Non-zero values of the matrix A in ELLPACK format.
    indices : 2D array of int
        Column indices corresponding to the non-zero values in data.
    u : 1D array of float
        Current solution vector.
    u_old : 1D array of float
        The solution vector at the previous time step.
    rhs : 1D array of float
        Right-hand side vector.
    u_old : 1D array of float
        The solution vector at the previous time step.
    indexes : 1D array of int
        Array of indexes where the solution is defined.
    dt : float
        Time step size.

    Returns
    -------
    np.ndarray
        Updated solution vector after the Forward Euler step.
    """
    u[indexes] = mx.sum(data * u_old[indexes][indices], axis=1) + dt * rhs[indexes]
    return u


def cg_mlx(indices, data, b, x0=None, indexes=None, atol=1e-8, maxiter=100):
    """
    Conjugate Gradient solver for Ax = b where A is represented in ELLPACK format.
    
    Parameters
    ----------
    data : 2D array of float
        Non-zero values of the matrix A in ELLPACK format.
    indices : 2D array of int
        Column indices corresponding to the non-zero values in data.
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

    if x0 is not None:
        x = x0
    else:
        x = mx.zeros_like(b)

    if indexes is not None:
        x_out = x
        x = x[indexes]
    
    # Initial residual and direction
    # r = b - A@x
    r = b - mx.sum(data * x[indices], axis=1)
    p = r
    r_norm = mx.sum(r * r)
    
    for i in range(maxiter):

        x, r, p, r_norm = cg_step(data, indices, x, r, p, r_norm)
        
        # Convergence check: Only sync with CPU every 10-20 iterations
        if i % 10 == 0:
            mx.eval(r_norm)
            if mx.sqrt(r_norm) < atol:
                break
        
    if indexes is not None:
        x_out[indexes] = x
    else:
        x_out = x

    if i == maxiter - 1:
        return x_out, -1
        
    return x_out, i


@mx.compile
def cg_step(data, indices, x, r, p, r_norm):
    # 1. SpMV: Ap = A @ p
    Ap = mx.sum(data * p[indices], axis=1)
    
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
def matvec_mlx(indices, data, x, indexes):
    """
    Performs A @ x where A is represented in ELLPACK format.

    Parameters
    ----------
    data : 2D array of float
        Non-zero values of the matrix A in ELLPACK format.
    indices : 2D array of int
        Column indices corresponding to the non-zero values in data.
    x : 1D array of float
        Vector to multiply with the matrix.
    indexes : 1D array of int
        Array of indexes where the solution is defined.

    Returns
    -------
    np.ndarray (len(indexes),)
        Result of the matrix-vector product.
    """
    return mx.sum(data * x[indexes][indices], axis=1)
