import numpy as np


class SimulationBackend:
    def __init__(self):
        self.name = "numpy"
        self.lib = np
        self.float_dtype = np.float64
        self.int_dtype = np.int64
        self.sparse_support = True

    def wrap(self, arr):
        if hasattr(arr, "__array_namespace__") and arr.size > 1:
            return self.lib.array(arr, dtype=self.float_dtype)

        return arr
    
    def wrap_indexes(self, arr):
        return self.lib.array(arr, dtype=self.int_dtype)
    
    def select_values(self, arr, inds):
        return arr.flat[inds]
    
    def set_values(self, arr, inds, values):
        arr[inds] = values
        return arr
    
    def set_flat_values(self, arr, inds, values):
        inds = np.atleast_1d(inds)
        arr.flat[inds] = values
        return arr

    def add_flat_values(self, arr, inds, values):
        arr.flat[inds] += values
        return arr

    def copy(self, arr):
        return arr.copy()


class NumbaBackend(SimulationBackend):
    def __init__(self):
        super().__init__()
        import numpy as np
        self.name = "numba"
        self.lib = np
        self.float_dtype = np.float64
        self.int_dtype = np.int64
        self.sparse_support = True


class MlxBackend(SimulationBackend):
    def __init__(self):
        super().__init__()
        import mlx.core as mx
        self.name = "mlx"
        self.lib = mx
        self.float_dtype = mx.float32  # MLX performs better with float32
        self.int_dtype = mx.int32
        self.sparse_support = False    # MLX does not support sparse matrices

    def select_values(self, arr, inds):
        return arr[inds]
    
    def set_values(self, arr, inds, values):
        arr[inds] = values
        return arr

    def set_flat_values(self, arr, inds, values):
        arr[inds] = values
        return arr
    
    def add_flat_values(self, arr, inds, values):
        arr[inds] += values
        return arr
    
    def copy(self, arr):
        return self.lib.array(arr, dtype=self.float_dtype)


class JaxBackend(SimulationBackend):
    def __init__(self):
        super().__init__()
        import jax.numpy as jnp
        self.name = "jax"
        self.lib = jnp
        self.float_dtype = jnp.float32 # JAX performs better with float32
        self.int_dtype = jnp.int32
        self.sparse_support = False    # JAX does not support sparse matrices

    def select_values(self, arr, inds):
        return arr[inds]
    
    def set_values(self, arr, inds, values):
        arr = arr.at[inds].set(values)
        return arr

    def set_flat_values(self, arr, inds, values):
        arr = arr.at[inds].set(values)
        return arr
    
    def add_flat_values(self, arr, inds, values):
        arr = arr.at[inds].add(values)
        return arr
    
    def copy(self, arr):
        return self.lib.array(arr, dtype=self.float_dtype, copy=True)
