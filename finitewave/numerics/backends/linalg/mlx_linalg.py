import mlx.core as mx
import numpy as np


@mx.compile
def axmy(a, x, y, indexes, out):
    """
    Performs the axmy operation:
    out[indexes] = a * x[indexes] - y[indexes]

    Parameters
    ----------
    a : float
        Scalar multiplier.
    x : 1D array of float
        Vector to be scaled.
    y : 1D array of float
        Vector to be subtracted.
    indexes : 1D array of int
        Array of indexes where the solution is defined.
    out : 1D array of float
        Output vector to store the result.
    
    Returns
    -------
    np.ndarray
        Updated solution vector after the axmy operation.
    """
    out = a * x - y
    return out


@mx.compile
def axpy(a, x, y, indexes, out):
    """
    Performs the axpy operation:
    out[indexes] = a * x[indexes] + y[indexes]

    Parameters
    ----------
    a : float
        Scalar multiplier.
    x : 1D array of float
        Vector to be scaled.
    y : 1D array of float
        Vector to be added.
    indexes : 1D array of int
        Array of indexes where the solution is defined.
    out : 1D array of float
        Output vector to store the result.

    Returns
    -------
    np.ndarray
        Updated solution vector after the axpy operation.
    """
    out = a * x + y
    return out


@mx.compile
def matvec(indices, data, x, indexes, out):
    """
    Performs A @ x where A is represented in ELLPACK format.

    Parameters
    ----------
    indices : 2D array of int
        Column indices corresponding to the non-zero values in data.
        (global to the myocyte indexes, e.g. x[indices])
    data : 2D array of float
        Non-zero values of the matrix A in ELLPACK format.
    x : 1D array of float
        Vector to multiply with the matrix.
    indexes : 1D array of int
        Array of indexes where the solution is defined.

    Returns
    -------
    np.ndarray (len(indexes),)
        Result of the matrix-vector product.
    """
    out = mx.sum(data * x[indices], axis=1)
    return out

@mx.compile
def matvec_p_ay(indices, data, x, y, a, indexes, out):
    """
    Performs the operation:
    out[indexes] = a * (data * x[indices]) + y[indexes]

    Parameters
    ----------
    data : 2D array of float
        Non-zero values of the matrix A in ELLPACK format.
    indices : 2D array of int
        Column indices corresponding to the non-zero values in data.
    x : 1D array of float
        Vector to multiply with the matrix.
    y : 1D array of float
        Vector to be added.
    a : float
        Scalar multiplier.
    indexes : 1D array of int
        Array of indexes where the solution is defined.
    out : 1D array of float
        Output vector to store the result.

    Returns
    -------
    np.ndarray
        Updated solution vector after the operation.
    """
    out = mx.sum(data * x[indices], axis=1) + a * y
    return out

def cg(indices, data, b, x0, indexes, atol=1e-8, maxiter=1):
    """
    Conjugate Gradient solver for Ax = b where A is represented in ELLPACK format.
    
    Parameters
    ----------
    indices : 2D array of int
        Column indices corresponding to the non-zero values in data.
        (local to the myocyte indexes, e.g. x0[indexes][indices])
    data : 2D array of float
        Non-zero values of the matrix A in ELLPACK format.
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
    x = x0[indexes]
    r = b[indexes] - mx.sum(data * x[indices], axis=1)
    p = r
    r_norm = mx.sum(r * r)

    for i in range(maxiter):

        x, r, p, r_norm = cg_step(data, indices, x, r, p, r_norm)
        
        # Convergence check: Only sync with CPU every 10-20 iterations
        if i % 10 == 0:
            mx.eval(r_norm)
            mx.synchronize()
            if mx.sqrt(r_norm) < atol:
                break
        
    x0[indexes] = x
        
    return x0, i


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



