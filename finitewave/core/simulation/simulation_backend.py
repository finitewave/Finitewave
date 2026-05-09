import warnings


class SimulationBackend:
    def __init__(self):
        import numpy as np
        self.name = "numpy"
        self.lib = np
        self.float_dtype = np.float64
        self.int_dtype = np.int64
        self.sparse_support = True

    def config(self, *args, **kwargs):
        """
        Configures the backend with the specified number of threads.
        """
        pass

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
        inds = self.lib.atleast_1d(inds)
        arr.flat[inds] = values
        return arr

    def add_flat_values(self, arr, inds, values):
        arr.flat[inds] += values
        return arr

    def copy(self, arr):
        return arr.copy()


class NumbaBackend(SimulationBackend):
    def __init__(self):
        import numpy as np
        self.name = "numba"
        self.lib = np
        self.float_dtype = np.float64
        self.int_dtype = np.int64
        self.sparse_support = True

    def config(self, num_of_threads, *args, **kwargs):
        """
        Sets the number of threads for Numba parallel operations.

        Parameters
        ----------
        num_of_threads : int or None
            The number of threads to use for Numba parallel operations.
            If None, it will use the maximum available threads minus one
            to avoid overloading the system.
        """
        import numba

        max_num_of_threads = numba.config.NUMBA_NUM_THREADS

        if num_of_threads is None:
            num_of_threads = max(1, max_num_of_threads - 1)

        if num_of_threads > max_num_of_threads:
            warnings.warn(
                f"({num_of_threads}) exceeds the available threads ({max_num_of_threads}). "
                f"Using the maximum available threads instead."
            )
            num_of_threads = min(num_of_threads, max_num_of_threads)

        numba.set_num_threads(num_of_threads)


class MlxBackend(SimulationBackend):
    def __init__(self):
        import mlx.core as mx
        self.name = "mlx"
        self.lib = mx
        self.float_dtype = mx.float32  # MLX performs better with float32
        self.int_dtype = mx.int32
        self.sparse_support = False    # MLX does not support sparse matrices

    @property
    def float_dtype(self):
        return self._float_dtype
    
    @float_dtype.setter
    def float_dtype(self, value):
        if value == self.lib.float64 or value == "float64":
            raise ValueError("MLX does not support float64. Please use float32.")
        
        if value == "float32":
            value = self.lib.float32

        self._float_dtype = value

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
        import jax.numpy as jnp
        self.name = "jax"
        self.lib = jnp
        self.float_dtype = jnp.float32 # JAX performs better with float32
        self.int_dtype = jnp.int32
        self.sparse_support = False    # JAX does not support sparse matrices

    @property
    def float_dtype(self):
        return self._float_dtype
    
    @float_dtype.setter
    def float_dtype(self, value):
        import jax
        self._float_dtype = value
        if value == self.lib.float64 or value == "float64":
            warnings.warn(
                "Using float64 with JAX may lead to slower performance. "
                "Consider using float32 for better performance."
            )
            jax.config.update("jax_enable_x64", True)

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
