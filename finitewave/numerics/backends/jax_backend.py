import warnings

from finitewave.core.backend.backend import Backend


class JaxBackend(Backend):
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