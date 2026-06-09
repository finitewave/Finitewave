import warnings


class SimulationBackend:
    def __init__(self):
        import numpy as np
        self.name = "numpy"
        self.lib = np
        self.float_dtype = np.float64
        self.int_dtype = np.int64
        self.sparse_support = True
        self.gpu_support = False

    def config(self, device=None, float_dtype=None, num_of_threads=None):
        """
        Configures the backend with specific settings. 
        This method should be overridden by subclasses to implement backend-specific configuration options.
        """
        pass

    def device_info(self):
        """
        Returns information about the computational device being used.
        """
        pass

    def wrap(self, arr):
        """
        Wraps an array into the backend's array type if it is not already.
        """
        # if hasattr(arr, "__array_namespace__") and arr.size > 1: - do lists have this?
        #     return self.lib.array(arr, dtype=self.float_dtype)
        if isinstance(arr, (int, float)):
            return arr
        return self.lib.array(arr, dtype=self.float_dtype)
    
    def wrap_indexes(self, arr):
        """
        Wraps an array of indexes into the backend's array type if it is not already.
        """
        return self.lib.array(arr, dtype=self.int_dtype)
    
    def select_values(self, arr, inds):
        """
        Selects values from an array based on the provided indexes.
        """
        return arr.flat[inds]
    
    def set_values(self, arr, inds, values):
        """
        Sets values in an array at the specified indexes.
        """
        arr[inds] = values
        return arr
    
    def set_flat_values(self, arr, inds, values):
        """
        Sets values in a flattened array at the specified indexes.
        """
        inds = self.lib.atleast_1d(inds)
        arr.flat[inds] = values
        return arr

    def add_flat_values(self, arr, inds, values):
        """
        Adds values to a flattened array at the specified indexes.
        """
        arr.flat[inds] += values
        return arr

    def copy(self, arr):
        """
        Creates a copy of the array.
        """
        return arr.copy()


class NumbaBackend(SimulationBackend):
    def __init__(self):
        import numpy as np
        self.name = "numba"
        self.lib = np
        self.float_dtype = np.float64
        self.int_dtype = np.int64
        self.sparse_support = True
        self.gpu_support = False

    def config(self, device=None, float_dtype=None, num_of_threads=None):
        """
        Sets the number of threads for Numba parallel operations.

        Parameters
        ----------
        num_of_threads : int or None
            The number of threads to use for Numba parallel operations.
            If None, it will use the maximum available threads minus one
            to avoid overloading the system.
        """
        if device not in (None, "cpu"):
            raise ValueError("Numba backend supports only device='cpu'.")

        if num_of_threads is None:
            return

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

    def device_info(self):
        return "cpu"


class MlxBackend(SimulationBackend):
    def __init__(self):
        import mlx.core as mx
        self.name = "mlx"
        self.lib = mx
        self.float_dtype = mx.float32  # MLX GPU supports only float32 (float64 is only for CPU)
        self.int_dtype = mx.int32
        self.sparse_support = False    # MLX does not support sparse matrices
        self.gpu_support = True

    def config(self, device=None, float_dtype=None, num_of_threads=None):
        import mlx.core as mx

        if device is not None:
            if device == "gpu":
                mx.set_default_device(mx.gpu)
                self.device = "gpu"
            elif device == "cpu":
                mx.set_default_device(mx.cpu)
                self.device = "cpu"
            else:
                raise ValueError("MLX device must be 'cpu' or 'gpu'.")

        if float_dtype is not None:
            self.float_dtype = float_dtype

    def device_info(self):
        import mlx.core as mx
        return mx.default_device()

    @property
    def float_dtype(self):
        return self._float_dtype
    
    @float_dtype.setter
    def float_dtype(self, value):
        if value == "float32":
            value = self.lib.float32
        elif value == "float64":
            value = self.lib.float64

        if value == self.lib.float64 and getattr(self, "device", "gpu") == "gpu":
            raise ValueError(
                "MLX float64 arrays only work with CPU operations. "
                "Use float32 for MLX GPU or set device='cpu'."
            )

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
        self.gpu_support = True        # possible if a specific package is installed (jaxlib with CUDA support), 
                                       # but not guaranteed in all environments 

    def config(self, device=None, float_dtype=None, num_of_threads=None):
        """
        Configures the JAX backend to use the specified device.
        Important: JAX device configuration must be done at the very beginning of the program, 
        before any JAX array is created.
        """
        import jax

        if device is not None:
            if device == "cpu":
                jax.config.update("jax_platform_name", "cpu")
            elif device == "gpu":
                jax.config.update("jax_platform_name", "gpu")
            else:
                raise ValueError("JAX device must be 'cpu' or 'gpu'.")

        if float_dtype is not None:
            self.float_dtype = float_dtype

    def device_info(self):
        import jax
        return jax.devices()

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
