import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def explicit_step(A, x, a, y, active_indexes, out):
    """
    Performs the operation:
    out[active_indexes] = A @ x[active_indexes] + a * y[active_indexes]

    Parameters
    ----------
    A : tuple
        A tuple containing (indices, data) representing the matrix A in ELLPACK format.
    x : 1D array of float
        Input vector to be multiplied by A.
    a : float
        Scalar multiplier.
    y : 1D array of float
        Input vector to be scaled and added.
    active_indexes : 1D array of int
        Indexes corresponding to active cells.
    out : 1D array of float
        Output vector to store the result.

    Returns
    -------
    np.ndarray
        The result of the operation.

    """
    y_local = y[active_indexes]
    out = out.at[active_indexes].set(matvec_ellpack(A, x) + a * y_local)
    return out

@jax.jit
def explicit_step_half_lumped(A_x, x, A_y, y, active_indexes, out):
    """
    Performs the operation:
    out[active_indexes] = A_x @ x[active_indexes] + A_y @ y[active_indexes]

    Parameters
    ----------
    A_x : tuple
        A tuple containing (indices, data) representing the matrix A_x in ELLPACK format.
    x : 1D array of float
        Input vector to be multiplied by A_x.
    A_y : tuple
        A tuple containing (indices, data) representing the matrix A_y in ELLPACK format.
    y : 1D array of float
        Input vector to be multiplied by A_y.
    active_indexes : 1D array of int
        Indexes corresponding to active cells.
    out : 1D array of float
        Output vector to store the result.

    Returns
    -------
    np.ndarray
        The result of the operation.
    """
    out = out.at[active_indexes].set(matvec_ellpack(A_x, x) + matvec_ellpack(A_y, y))
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
    A : tuple
        A tuple containing (indices, data, indexes) representing the matrix in ELLPACK format.
    x : 1D array of float
        Input vector to be multiplied by A.

    Returns
    -------
    np.ndarray
        Result of the matrix-vector multiplication A @ x.
    """
    indices, data = A
    out = jnp.sum(data * x[indices], axis=1)
    return out
