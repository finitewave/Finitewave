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
        u = u_old.at[indexes].set(jnp.sum(data * u_old[indices], axis=1) + 
                                  dt * rhs[indexes])
        return u