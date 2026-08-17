from abc import ABC, abstractmethod


def axmy(a, x, y, indexes, out):
    """Performs the operation out = a * x + y for the specified indexes.

    Parameters
    ----------
    a : float
        Scalar multiplier for x.
    x : backend-specific array
        The input vector to be scaled and added.
    y : backend-specific array
        The input vector to be added.
    indexes : 1D array of int
        Array of indexes where the operation is performed.
    out : backend-specific array
        The output vector to store the result.

    Returns
    -------
    Updated output vector after performing the axmy operation.
    """
    raise NotImplementedError("axmy must be implemented by subclasses.")
    

def axpy(a, x, y, indexes, out):
    """Performs the operation out = a * x + y for the specified indexes.

    Parameters
    ----------
    a : float
        Scalar multiplier for x.
    x : backend-specific array
        The input vector to be scaled and added.
    y : backend-specific array
        The input vector to be added.
    indexes : 1D array of int
        Array of indexes where the operation is performed.
    out : backend-specific array
        The output vector to store the result.

    Returns
    -------
    Updated output vector after performing the axpy operation.
    """
    raise NotImplementedError("axpy must be implemented by subclasses.")


def matvec(A, x, indexes, out):
    """Performs the matrix-vector multiplication out = A @ x where A is represented in a backend-specific format.

    Parameters
    ----------
    A : tuple
        A representation of the matrix A in a backend-specific format (e.g., ELLPACK, CSR).
    x : backend-specific array
        The input vector to be multiplied by the matrix.
    indexes : 1D array of int
        Array of indexes where the operation is performed.
    out : backend-specific array
        The output vector to store the result.

    Returns
    -------
    Updated output vector after performing the matrix-vector multiplication.
    """
    raise NotImplementedError("matvec must be implemented by subclasses.")


def matvec_p_ay(A, x, y, a, indexes, out):
    """Performs the operation out = A @ x + a * y where A is represented in a backend-specific format.

    Parameters
    ----------
    A : tuple
        A representation of the matrix A in a backend-specific format (e.g., ELLPACK, CSR).
    x : backend-specific array
        The input vector to be multiplied by the matrix.
    y : backend-specific array
        The input vector to be scaled and added.
    a : float
        Scalar multiplier for y.
    indexes : 1D array of int
        Array of indexes where the operation is performed.
    out : backend-specific array
        The output vector to store the result.

    Returns
    -------
    Updated output vector after performing the matrix-vector multiplication and addition.
    """
    raise NotImplementedError("matvec_p_ay must be implemented by subclasses.")

def cg(A, b, x0, M=None, atol=1e-8, maxiter=1):
    """Solves the linear system Ax = b using the Conjugate Gradient method, where A is represented in a backend-specific format.

    Parameters
    ----------
    A : tuple
        A representation of the matrix A in a backend-specific format (e.g., ELLPACK, CSR).
    b : backend-specific array
        The right-hand side vector.
    x0 : backend-specific array
        The initial guess for the solution.
    M : tuple, optional
        A representation of the preconditioner matrix in a backend-specific format (e.g., ELLPACK, CSR).
    atol : float, optional
        Absolute tolerance for convergence (default is 1e-8).
    maxiter : int, optional
        Maximum number of iterations (default is 1).

    Returns
    -------
    Tuple containing:
        - Updated solution vector after solving Ax = b.
        - Number of iterations taken to converge.
    """
    raise NotImplementedError("cg must be implemented.")

