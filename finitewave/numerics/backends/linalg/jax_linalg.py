import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def explicit_step(A, x, y, a, out):
    """
    Performs the operation:
    out[indexes] = A @ x[indices] + a * y[indexes]

    Parameters
    ----------
    A : tuple
        A tuple containing (indices, data) representing the matrix A in ELLPACK format.
    x : 1D array of float
        Input vector to be multiplied by A.
    y : 1D array of float
        Input vector y.
    a : float
        Scalar multiplier.
    indexes : 1D array of int
        Array of indexes where the solution is defined.
    out : 1D array of float
        Output vector to store the result.

    Returns
    -------
    np.ndarray

    """
    indices, data, indexes = A
    x_local = x[indexes]
    y_local = y[indexes]
    out = out.at[indexes].set(matvec_ellpack(A, x_local) + a * y_local)
    return out

@jax.jit
def explicit_step_half_lumped(A_x, x, y, A_y, out):
    indices_x, data_x, indexes_x = A_x
    indices_y, data_y, indexes_y = A_y
    x_local = x[indexes_x]
    y_local = y[indexes_y]
    out = out.at[indexes_x].set(matvec_ellpack(A_x, x_local) + matvec_ellpack(A_y, y_local))
    return out
    


def implict_step(method, A_lhs, A_rhs, A_ion, x_old, x_old_2, i_ion, active_indexes, out, **kwargs):
    """
    Implicit solver for the diffusion equation using the Conjugate Gradient method.

    Parameters
    ----------
    method : str
        The method to use for solving the linear system. Currently supports 'cg' (Conjugate Gradient).
    A_lhs : tuple
        Left-hand side matrix in ELLPACK format.
    A_rhs : tuple
        Right-hand side matrix in ELLPACK format.
    A_ion : tuple
        Ionic contribution matrix in ELLPACK format.
    x_old : 1D array of float
        Previous solution vector.
    x_old_2 : 1D array of float
        Solution vector from two time steps ago.
    i_ion : 1D array of float
        Ionic current vector.
    active_indexes : 1D array of int
        Indexes corresponding to active cells.
    fibro_mask : 1D array of bool
        Mask indicating fibroblast cells.
    dt : float
        Time step size.
    order : int
        Order of the time integration method (1 or 2).

    Returns
    -------
    x_new : 1D array of float
        Updated solution vector after the implicit step.
    """
    x_old_local = x_old[active_indexes]
    x_old_2_local = x_old_2[active_indexes]
    i_ion_local = i_ion[active_indexes]

    x0 = 2. * x_old_local - x_old_2_local
    b = matvec_ellpack(A_rhs, x_old_local) + matvec_ellpack(A_ion, i_ion_local)

    A_matvec = lambda x: matvec_ellpack(A_lhs, x)
    x_local, info = method(A_matvec, b, x0, **kwargs)
    out = out.at[active_indexes].set(x_local)
    return out, info


@jax.jit
def matvec_ellpack(A, x):
    """
    Performs A @ x where A is represented in ELLPACK format.

    Parameters
    ----------
    indices : 2D array of int
        Column indices corresponding to the non-zero values in data.
    data : 2D array of float
        Non-zero values of the matrix A in ELLPACK format.
    x : 1D array of float
        Input vector to be multiplied by A.
    indexes : 1D array of int
        Array of indexes where the solution is defined.

    Returns
    -------
    np.ndarray
        Result of the matrix-vector multiplication A @ x.
    """
    indices, data, indexes = A
    out = jnp.sum(data * x[indices], axis=1)
    return out


@jax.jit
def cg(A, b, x0, M=None, atol=1e-8, maxiter=100):
    """
    Conjugate Gradient solver for Ax = b where A is represented in ELLPACK format.
    
    Parameters
    ----------
    A : tuple
        A tuple containing (indices, data) representing the matrix A in ELLPACK format.
        Indices must be in local indexing.
    b : 1D array of float
        Right-hand side vector.
    x0 : 1D array of float
        Initial guess for the solution.
    M : tuple, optional
        Preconditioner matrix in ELLPACK format.
    atol : float
        Absolute tolerance for convergence.
    maxiter : int
        Maximum number of iterations.
        
    Returns
    -------
    x : 1D array of float
        Approximate solution vector.
    int
        Number of iterations performed, or -1 if not converged within maxiter.
    """
    indices, data, indexes = A
    # Initial state
    r0 = b[indexes] - matvec_ellpack(A, x0[indexes])
    p0 = r0
    r_norm0 = jnp.sum(r0 * r0)
    
    # We pack everything that changes into a "state" tuple
    initial_state = (0, x0[indexes], r0, p0, r_norm0)
    
    def is_not_converged(state):
        i, _, _, _, r_norm = state
        # Continue loop if we haven't hit maxiter AND error is too high
        return (i < maxiter) & (jnp.sqrt(r_norm) > atol)

    def cg_step(state):
        i, x, r, p, r_norm = state
        
        # 1. A @ p
        Ap = matvec_ellpack(A, p)
        
        # 2. r @ r / p @ Ap
        alpha = r_norm / jnp.sum(p * Ap)
        
        # 3. Updates
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        
        # 4. Beta
        r_norm_new = jnp.sum(r_new * r_new)
        p_new = r_new + (r_norm_new / r_norm) * p
        
        return (i + 1, x_new, r_new, p_new, r_norm_new)

    i, x, _, _, _ = jax.lax.while_loop(is_not_converged, cg_step, initial_state)

    x = x0.at[indexes].set(x)

    return x, i
