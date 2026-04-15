import mlx.core as mx


def forward_euler_mlx(indices, data, u, u_new, rhs, indexes, dt):
    """
    Performs the Forward Euler step:
    y = u + M^{-1} * (A * u + rhs)

    Parameters
    ----------
    data : 2D array of float
        Non-zero values of the matrix A in ELLPACK format.
    indices : 2D array of int
        Column indices corresponding to the non-zero values in data.
    u : 1D array of float
        Current solution vector.
    rhs : 1D array of float
        Right-hand side vector.
    u_new : 1D array of float
        Array to store the updated solution vector.
    indexes : 1D array of int
        Array of indexes where the solution is defined.
    dt : float
        Time step size.

    Returns
    -------
    np.ndarray
        Updated solution vector after the Forward Euler step.
    """
    u_new = u - dt * mx.sum(data * u[indices], axis=1) + dt * rhs
    return u_new


def cg_mlx(indices, data, b, x0=None, indexes=None, atol=1e-8, max_iter=100):
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
    max_iter : int, optional
        Maximum number of iterations.
        
    Returns
    -------
    x : 1D array of float
        Approximate solution vector.
    int
        Number of iterations performed, or -1 if not converged within max_iter.
    """

    if x0 is not None:
        x = x0
    else:
        x = mx.zeros_like(b)

    if indexes is not None:
        x_out = x
        x = x[indexes]
        b = b[indexes]
    
    # Initial residual and direction
    # r = b - A@x
    r = b - mx.sum(data * x[indices], axis=1)
    p = r
    rsold = mx.sum(r * r)
    
    for i in range(max_iter):

        x, r, p, rsnew = cg_step(data, indices, x, r, p, rsold)
        
        # Convergence check: Only sync with CPU every 10-20 iterations
        if i % 10 == 0:
            mx.eval(rsnew)
            if mx.sqrt(rsnew) < atol:
                break
        
        rsold = rsnew

    if indexes is not None:
        x_out[indexes] = x
    else:
        x_out = x
        
    return x_out, i


def cg_step(data, indices, x, r, p, rsold):
    # 1. SpMV: Ap = A @ p
    Ap = mx.sum(data * p[indices], axis=1)
    
    # 2. Alpha: step size
    alpha = rsold / mx.sum(p * Ap)
    
    # 3. Update solution and residual
    x = x + alpha * p
    r = r - alpha * Ap
    
    # 4. Beta: direction update
    rsnew = mx.sum(r * r)
    p = r + (rsnew / rsold) * p
    
    return x, r, p, rsnew
