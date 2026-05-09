import mlx.core as mx
import numpy as np


class MlxMethod:
    def __init__(self):
        pass

    def wrap_matrix(self, csr_matrix, dtype, indexes=None):
        """Converts a sparse matrix in CSR format to ELLPACK format.

        Parameters
        ----------
        csr_matrix : scipy.sparse.csr_matrix
            The input sparse matrix in CSR format.
        dtype : np.dtype
            The data type for the ELLPACK format arrays.
        indexes : 1D array of int, optional
            Array of indexes where the solution is defined.

        Returns
        -------
        indices : mx.ndarray
            The column indices of the non-zero elements in ELLPACK format.
        data : mx.ndarray
            The non-zero values of the matrix in ELLPACK format.
        """
        row_lengths = np.diff(csr_matrix.indptr)
        n_cols = np.max(row_lengths)
        n_rows = csr_matrix.shape[0]

        ellpack_indices = np.repeat(np.arange(n_rows), n_cols).reshape(n_rows, n_cols)
        ellpack_data = np.zeros((n_rows, n_cols), dtype=np.float32)

        inds = np.repeat([np.arange(n_cols)], n_rows, axis=0)
        mask = inds < row_lengths[:, None]
        ellpack_indices[mask] = csr_matrix.indices
        ellpack_data[mask] = csr_matrix.data.astype(np.float32)

        ellpack_indices = mx.array(ellpack_indices, dtype=mx.int32)
        ellpack_data = mx.array(ellpack_data, dtype=dtype)

        if indexes is not None:
            indexes = mx.array(indexes, dtype=mx.int32)
            ellpack_indices = indexes[ellpack_indices]

        return ellpack_indices, ellpack_data
    

class MlxEuler(MlxMethod):
    def __init__(self):
        pass

    @staticmethod
    def evaluate(u, step):
        if step % 10 == 0:
            mx.eval(u)
        return u

    @staticmethod
    @mx.compile
    def solve(indices, data, u_old, rhs, dt, indexes, u):
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
        u[indexes] = mx.sum(data * u_old[indices], axis=1) + dt * rhs[indexes]
        return u


class MlxCG(MlxMethod):
    def __init__(self):
        pass

    @staticmethod
    @mx.compile
    def axmy(a, x, y, indexes, out):
        out[indexes] = a * x[indexes] - y[indexes]
        return out

    @staticmethod
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
        out[indexes] = a * x[indexes] + y[indexes]
        return out
    
    @staticmethod
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
        out[indexes] = mx.sum(data * x[indexes][indices], axis=1)
        return out
    
    @staticmethod
    def solve(indices, data, b, x0, indexes, atol=1e-8, maxiter=1):
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



