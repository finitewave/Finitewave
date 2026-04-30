import jax
import jax.numpy as jnp


class JaxEuler:
    def __init__(self):
        pass

    @staticmethod
    def evaluate(u, step):
        return u
    
    @staticmethod
    @jax.jit
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
        u = u_old.at[indexes].set(jnp.sum(data * u_old[indices], axis=1) + 
                                  dt * rhs[indexes])
        return u
    

class JaxCG:
    def __init__(self):
        pass
    
    @staticmethod
    @jax.jit
    def axmy(a, x, y, indexes, out):
        out = out.at[indexes].set(a * x[indexes] - y[indexes])
        return out

    @staticmethod
    @jax.jit
    def axpy(a, x, y, indexes, out):
        """
        Performs the axpy operation:
        out[indexes] = a * x[indexes] + y[indexes]

        Parameters
        ----------

        a : float
            Scalar multiplier.
        x : 1D array of float
            Input vector x.
        y : 1D array of float
            Input vector y.
        indexes : 1D array of int
            Array of indexes where the operation is applied.
        out : 1D array of float
            Output vector to store the result.

        Returns
        -------
        np.ndarray
            Updated output vector after the axpy operation.
        """
        out = out.at[indexes].set(a * x[indexes] + y[indexes])
        return out
    
    @staticmethod
    @jax.jit
    def matvec(indices, data, x, indexes, out):
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
        out = out.at[indexes].set(jnp.sum(data * x[indexes][indices], axis=1))
        return out

    @staticmethod
    @jax.jit
    def solve(indices, data, b, x0, indexes, atol=1e-8, maxiter=100):
        """
        Conjugate Gradient solver for Ax = b where A is represented in ELLPACK format.
        
        Parameters
        ----------
        indices : 2D array of int
            Column indices corresponding to the non-zero values in data.
        data : 2D array of float
            Non-zero values of the matrix A in ELLPACK format.
        b : 1D array of float
            Right-hand side vector.
        x0 : 1D array of float
            Initial guess for the solution.
        indexes : 1D array of int
            Array of indexes where the solution is defined.
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
        # Initial state
        r0 = b[indexes] - jnp.sum(data * x0[indexes][indices], axis=1)
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
            Ap = jnp.sum(data * p[indices], axis=1)
            
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