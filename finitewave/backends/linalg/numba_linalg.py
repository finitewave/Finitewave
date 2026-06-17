import numpy as np
from numba import njit, prange

from finitewave.core.backend.linalg.backend_linalg import BackendLinalg


@njit(parallel=True, fastmath=True, cache=True)
def matvec_p_ay_numba(indptr, indices, data, x, y, a, indexes, out):
    """
    Performs the Forward Euler step:
    x_new = A @ x + a * y

    Parameters
    ----------
    indptr : np.ndarray
        CSR index pointer array.
    indices : np.ndarray
        CSR indices array.
    data : np.ndarray
        CSR data array.
    x : np.ndarray
        Input vector to be multiplied by A.
    y : np.ndarray
        Input vector to be scaled by a and added to A @ x.
    a : float
        Scalar multiplier for y.
    indexes : np.ndarray
        Array of indexes where the solution is defined.
    out : np.ndarray
        Output solution vector.

    Returns
    -------
    out : np.ndarray
        Vector with the result of A @ x + a * y for the given indexes.
    """
    n_rows = indptr.size - 1

    for i in prange(n_rows):
        start, end = indptr[i], indptr[i+1]
        if start == end:
            continue
        a_x = 0.0
        for j in range(start, end):
            jj = indices[j]
            jj = indexes[jj]
            a_x += data[j] * x.flat[jj]

        ii = indexes[i]
        out.flat[ii] = a_x + a * y.flat[ii]

    return out


@njit(parallel=True, fastmath=True, cache=True)
def matvec_numba(indptr, indices, data, x, out, indexes):
    """
    Computes out = A @ x for a sparse matrix A in CSR format.
    """
    n = indptr.shape[0] - 1
    for i in prange(n):
        start, end = indptr[i], indptr[i+1]
        if start == end:
            continue
        out_i = 0.
        for j in range(start, end):
            jj = indices[j]
            jj = indexes[jj]
            out_i += data[j] * x.flat[jj]

        ii = indexes[i]
        out.flat[ii] = out_i
    return out


@njit(parallel=True, fastmath=True, cache=True)
def matvec_and_dot_numba(indptr, indices, data, x, y, indexes):
    """Computes y = A @ x and returns the dot product x^T @ (A @ x).
    """
    n = indptr.shape[0] - 1
    s = 0.
    for i in prange(n):
        start, end = indptr[i], indptr[i+1]
        if start == end:
            continue
        y_i = 0.
        for j in range(start, end):
            jj = indices[j]
            jj = indexes[jj]
            y_i += data[j] * x.flat[jj]

        ii = indexes[i]
        y.flat[ii] = y_i
        s += y_i * x.flat[ii]
    return y, s


@njit(parallel=True, fastmath=True, cache=True)
def b_m_matvec_numba(indptr, indices, data, b, x, y, indexes):
    """
    Computes y = b - A @ x
    """
    n = indptr.shape[0] - 1
    for i in prange(n):
        start, end = indptr[i], indptr[i+1]
        if start == end:
            continue
        y_i = 0.
        for j in range(start, end):
            jj = indices[j]
            jj = indexes[jj]
            y_i += data[j] * x.flat[jj]

        ii = indexes[i]
        y.flat[ii] = b.flat[ii] - y_i
    return y


@njit(parallel=True, fastmath=True, cache=True)
def dot_numba(x, y, indexes):
    n = len(indexes)
    s = 0.
    for i in prange(n):
        ii = indexes[i]
        s += x.flat[ii] * y.flat[ii]
    return s


@njit(parallel=True, fastmath=True, cache=True)
def ax_m_y_numba(a, x, y, indexes, out):
    """Computes out = a * x - y in place.
    """
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        out.flat[ii] = a * x.flat[ii] - y.flat[ii]
    return out


@njit(parallel=True, fastmath=True, cache=True)
def ax_p_y_numba(a, x, y, indexes, out):
    """Computes out = a * x + y in place.
    """
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        out.flat[ii] = a * x.flat[ii] + y.flat[ii]
    return out


@njit(parallel=True, fastmath=True, cache=True)
def y_pm_ax_numba(a, xp, yp, xm, ym, indexes):
    """
    Computes yp = yp + a * xp and ym = ym - a * xm in place.
    """
    ym_dot_ym = 0.
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        yp.flat[ii] += a * xp.flat[ii]

        ym_i = ym.flat[ii] - a * xm.flat[ii]
        ym.flat[ii] = ym_i
        ym_dot_ym += ym_i * ym_i
    return yp, ym, ym_dot_ym


@njit(parallel=True, fastmath=True, cache=True)
def copyto_numba(x, y, indexes):
    """
    Copies x to y in place for the given indexes.
    """
    n = len(indexes)
    for i in prange(n):
        ii = indexes[i]
        y.flat[ii] = x.flat[ii]
    return y
    

class NumbaLinalg(BackendLinalg):
    def __init__(self):
        pass

    def wrap_matrix(self, crs_matrix, dtype, indexes=None):
        """Converts a sparse matrix in CRS format to NumPy arrays.

        Parameters
        ----------
        crs_matrix : scipy.sparse.csr_matrix
            The input sparse matrix in CRS format.
        dtype : np.dtype
            The data type for the output arrays.
        indexes : np.ndarray, optional
            Array of indexes where the solution is defined.

        Returns
        -------
        indptr : np.ndarray
            The index pointer array of the CRS format.
        indices : np.ndarray
            The column indices of the non-zero elements in CRS format.
        data : np.ndarray
            The non-zero values of the matrix in CRS format.
        """
        data = crs_matrix.data.astype(dtype)
        return crs_matrix.indptr, crs_matrix.indices, data

    @staticmethod
    def evaluate(u, step):
        return u
    
    @staticmethod
    def explicit_step(indptr, indices, data, u_old, rhs, dt, indexes, u):
        """Performs the Forward Euler step for the diffusion equation.

        Parameters
        ----------
        indptr : np.ndarray
            The index pointer array of the CRS format.
        indices : np.ndarray
            The column indices of the non-zero elements in CRS format.
        data : np.ndarray
            The non-zero values of the matrix in CRS format.
            The stiffness matrix representing the diffusion operator.
        u_old : np.ndarray
            Previous solution vector.
        rhs : np.ndarray
            Right-hand side vector (e.g., source term).
        dt : float
            Time step size.
        indexes : np.ndarray
            Array of indexes where the solution is defined.
        u : np.ndarray
            Output solution vector to store the updated solution after the Forward Euler step.

        Returns
        -------
        np.ndarray
            Updated solution vector after the Forward Euler step.
        """
        return matvec_p_ay_numba(indptr, indices, data, u_old, rhs, dt, indexes, u)

    @staticmethod
    def axmy(a, x, y, indexes, out):
        return ax_m_y_numba(a, x, y, indexes, out)

    @staticmethod
    def axpy(a, x, y, indexes, out):
        """Performs the axpy operation:
        out[indexes] = a * x[indexes] + y[indexes]

        Parameters
        ----------
        a : float
            Scalar multiplier for x.
        x : np.ndarray
            Input vector x.
        y : np.ndarray
            Input vector y.
        indexes : np.ndarray
            Array of indexes where the solution is defined.
        out : np.ndarray
            Output vector to store the result of the axpy operation.

        Returns
        -------
        np.ndarray
            Updated solution vector after the axpy operation.
        """
        return ax_p_y_numba(a, x, y, indexes, out)
    
    @staticmethod
    def matvec(indptr, indices, data, x, indexes, out):
        """Performs the matrix-vector multiplication for a sparse matrix in CRS format.

        Parameters
        ----------
        indptr : np.ndarray
            The index pointer array of the CRS format.
        indices : np.ndarray
            The column indices of the non-zero elements in CRS format.
        data : np.ndarray
            The non-zero values of the matrix in CRS format.
        x : np.ndarray
            Input vector to be multiplied by the matrix.
        indexes : np.ndarray
            Array of indexes where the solution is defined.
        out : np.ndarray
            Output vector to store the result of the matrix-vector multiplication.

        Returns
        -------
        np.ndarray
            Updated solution vector after the matrix-vector multiplication.
        """
        return matvec_numba(indptr, indices, data, x, out, indexes)
    
    @staticmethod
    def cg_solve(indptr, indices, data, b, x, indexes, rtol=None, atol=1e-6, maxiter=100):
        """ Conjugate Gradient solver for Ax = b for x, where A is a sparse
        matrix given in CSR format.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            The matrix A in Ax = b.
        b : 1D array of float
            Right-hand side vector.
        x : 1D array of float
            Initial guess for the solution, will be modified in place.
        indexes : 1D array of int
            Array of indexes where the solution is defined.
        atol : float
            Absolute tolerance for the stopping criterion.
        maxiter : int
            Maximum number of iterations.

        Returns
        -------
        x : 1D array of float
            Approximate solution vector.
        int
            Number of iterations performed, or -1 if not converged within maxiter.
        """
        b_norm_2 = dot_numba(b, b, indexes)

        if b_norm_2 == 0:
            return x, 0
        
        if rtol is not None:
            atol = max(atol, rtol * np.sqrt(b_norm_2))

        r = np.empty_like(x)
        r = b_m_matvec_numba(indptr, indices, data, b, x, r, indexes)
        q = np.empty_like(r)

        p = np.empty_like(r)
        p = copyto_numba(r, p, indexes)

        r_norm = dot_numba(r, r, indexes)

        if np.sqrt(r_norm) < atol:
            return x, 0

        for iteration in range(maxiter):

            q, p_dot_q = matvec_and_dot_numba(indptr, indices, data, p, q, indexes)

            alpha = r_norm / p_dot_q

            x, r, r_norm_new = y_pm_ax_numba(alpha, p, x, q, r, indexes)

            if np.sqrt(r_norm_new) < atol:
                return x, iteration

            beta = r_norm_new / r_norm
            p = ax_p_y_numba(beta, p, r, indexes, p)
            r_norm = r_norm_new

        return x, iteration

    @staticmethod
    def preconditioned_solve(indptr, indices, data, b, x, indexes, M, rtol=None, atol=1e-6, maxiter=100):
        # TODO: should be an argument to the cg_solve.
        """ Preconditioned Conjugate Gradient solver for A@x = b for x, where A is
        a sparse matrix in CSR format.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            The matrix A in Ax = b.
        b : 1D array of float
            Right-hand side vector.
        x : 1D array of float
            Initial guess for the solution, will be modified in place.
        indexes : 1D array of int
            Array of indexes where the solution is defined.
        M : object
            The preconditioner to use. Must have a method `matvec(r, z, indexes)`
            that applies the preconditioner to a vector r and stores the result in z.
        rtol : float
            Relative tolerance for convergence. The solver will stop when the
            residual norm is less than max(atol, rtol * ||b||).
            If None, only the absolute tolerance is used.
        atol : float
            Absolute tolerance for the stopping criterion.
        maxiter : int
            Maximum number of iterations.

        Returns
        -------
        x : 1D array of float
            Approximate solution vector.
        int
            Number of iterations performed, or -1 if not converged within maxiter.
        """
        precondtioner = M.matvec

        b_norm_2 = dot_numba(b, b, indexes)

        if b_norm_2 == 0:
            return x, 0
        
        if rtol is not None:
            atol = max(atol, rtol * np.sqrt(b_norm_2))

        
        r = np.empty_like(x)
        r = b_m_matvec_numba(indptr, indices, data, b, x, r, indexes)
        q = np.empty_like(r)
        z = np.empty_like(r)

        rho_prev, p = None, None

        for iteration in range(maxiter):

            r_norm_2 = dot_numba(r, r, indexes)
            if np.sqrt(r_norm_2) < atol:  # Are we done?
                return x, iteration
            
            z = precondtioner(r, z, indexes)
            rho_cur = dot_numba(r, z, indexes)

            if iteration > 0:
                beta = rho_cur / rho_prev
                p = ax_p_y_numba(beta, p, r, indexes, p)
            else:  # First spin
                p = np.empty_like(r)
                p[:] = z[:]

            q, p_dot_q = matvec_and_dot_numba(indptr, indices, data, p, q, indexes)
            alpha = rho_cur / p_dot_q
            rho_prev = rho_cur
            x, r, rho_cur = y_pm_ax_numba(alpha, p, x, q, r, indexes)

        return x, -1