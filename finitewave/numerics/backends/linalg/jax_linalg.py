from functools import partial
import jax
import jax.numpy as jnp
import numpy as np


def select_explicit_solver(x, active_indexes):
    if x.size == active_indexes.size:
        return explicit_step
    
    if x.size > active_indexes.size:
        return explicit_step_indexed

    raise ValueError("Invalid combination of x and active_indexes sizes.")


@jax.jit
def explicit_step(A_x, x, A_y, y, active_indexes, out):
    """
    Performs the operation:
    out = A_x @ x + A_y @ y

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
    return matvec_ellpack(A_x, x) + matvec_ellpack(A_y, y)


@jax.jit
def explicit_step_indexed(A_x, x, A_y, y, active_indexes, out):
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




@partial(jax.jit, static_argnames=('method',))
def implicit_step(method, A_lhs, A_rhs, A_ion, x_old, x_old_2, i_ion, active_indexes, out, **kwargs):
    """
    Implicit solver for the diffusion equation using the Conjugate Gradient method.

    Parameters
    ----------
    method : str
        The method to use for solving the linear system. Currently supports 'cg' (Conjugate Gradient).
    A_lhs : tuple
        Left-hand side matrix in ELLPACK format.
        Indexing should be local to the active indexes.
    A_rhs : tuple
        Right-hand side matrix in ELLPACK format.
        Indexing should be global to the entire domain.
    A_ion : tuple
        Ionic contribution matrix in ELLPACK format.
        Indexing should be global to the entire domain.
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
    x0 = 2. * x_old[active_indexes] - x_old_2[active_indexes]
    b = matvec_ellpack(A_rhs, x_old) + matvec_ellpack(A_ion, i_ion)

    A_matvec = lambda x: matvec_ellpack(A_lhs, x)
    x_local, info = method(A_matvec, b, x0, **kwargs)
    out = out.at[active_indexes].set(x_local)
    return out, info


@jax.jit
def prepare_implicit_step(A_rhs, A_ion, x_old, x_old_2, i_ion, active_indexes):
    """
    Prepares the initial guess and right-hand side for the implicit step.

    Parameters
    ----------
    A_rhs : tuple
        Right-hand side matrix in ELLPACK format.
        Indexing should be global to the entire domain.
    A_ion : tuple
        Ionic contribution matrix in ELLPACK format.
        Indexing should be global to the entire domain.
    x_old : 1D array of float
        Previous solution vector.
    x_old_2 : 1D array of float
        Solution vector from two time steps ago.
    i_ion : 1D array of float
        Ionic current vector.
    active_indexes : 1D array of int
        Indexes corresponding to active cells.

    Returns
    -------
    x0 : 1D array of float
        Initial guess for the implicit solver.
    b : 1D array of float
        Right-hand side vector for the implicit solver.
    """
    x0 = 2. * x_old[active_indexes] - x_old_2[active_indexes]
    b = matvec_ellpack(A_rhs, x_old) + matvec_ellpack(A_ion, i_ion)
    return x0, b


@jax.jit
def update_active_indexes(x, active_indexes, out):
    """
    Updates the output vector with the values from x at the specified active indexes.

    Parameters
    ----------
    x : 1D array of float
        Input vector containing updated values.
    active_indexes : 1D array of int
        Indexes corresponding to active cells.
    out : 1D array of float
        Output vector to be updated.

    Returns
    -------
    np.ndarray
        Updated output vector with values from x at the active indexes.
    """
    out = out.at[active_indexes].set(x)
    return out


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
