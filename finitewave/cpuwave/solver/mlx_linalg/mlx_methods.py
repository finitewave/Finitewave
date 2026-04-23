import mlx.core as mx


class MlxEuler:
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


class MlxCG:
    def __init__(self):
        pass

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
    def solve(indices, data, b, x0, indexes, atol=1e-8, maxiter=100):
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
        r = b[indexes] - mx.sum(data * x0[indexes][indices], axis=1)
        p = r
        r_norm = mx.sum(r * r)
        
        for i in range(maxiter):

            x, r, p, r_norm = cg_step(data, indices, x, r, p, r_norm)
            
            # Convergence check: Only sync with CPU every 10-20 iterations
            if i % 10 == 0:
                mx.eval(r_norm)
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



