import numpy as np
from .numba_linalg import (
    dot_numba,
    b_m_matvec_numba,
    ax_p_y_numba,
    matvec_and_dot_numba,
    matvec_p_ay_numba,
    y_pm_ax_numba,
    copyto_numba,)


class NumbaEuler:
    def __init__(self):
        pass

    @staticmethod
    def evaluate(u, step):
        return u
    
    @staticmethod
    def solve(indptr, indices, data, u_old, rhs, dt, indexes, u):
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


class NumbaCG():
    def __init__(self):
        pass
    
    @staticmethod
    def solve(indptr, indices, data, b, x, indexes, rtol=None, atol=1e-6, maxiter=100):
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

        r_dot_r = dot_numba(r, r, indexes)

        if np.sqrt(r_dot_r) < atol:
            return x, 0

        for iteration in range(maxiter):
            r_dot_r_prev = r_dot_r

            q, p_dot_q = matvec_and_dot_numba(indptr, indices, data, p, q, indexes)

            alpha = r_dot_r / p_dot_q
            x, r, r_dot_r = y_pm_ax_numba(alpha, p, x, q, r, indexes)

            if np.sqrt(r_dot_r) < atol:
                return x, iteration

            beta = r_dot_r / r_dot_r_prev
            p = ax_p_y_numba(beta, p, r, indexes, p)

        return x, -1


class PreconditionedCG(NumbaCG):
    @staticmethod
    def solve(indptr, indices, data, b, x, indexes, M, rtol=None, atol=1e-6, maxiter=100):
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
