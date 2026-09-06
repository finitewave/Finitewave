import warnings
import numpy as np
from functools import wraps
from .linalg.sparse import csr_to_ellpack

from finitewave.core.backend.backend import Backend


def _check_jax_installed():
    try:
        import jax
    except ImportError as exc:
        if exc.name == "jax" or exc.name.startswith("jax."):
            raise ImportError(
                "JAXBackend requires the optional 'jax' package."
            ) from exc
        raise


def _jax_jit(func):
    """
    Lazily JIT-compile a JaxBackend instance method.
    """
    compiled = None

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        nonlocal compiled

        if compiled is None:
            import jax
            compiled = jax.jit(func)

        return compiled(self, *args, **kwargs)

    return wrapper


class JAXBackend(Backend):
    def __init__(self):
        super().__init__()
        _check_jax_installed()
        import jax
        import jax.numpy as jnp
        from .linalg import jax_linalg
        from .model.jax_model_generator import JAXModelGenerator

        self.name = "jax"
        self.lib = jnp
        self.float_dtype = jnp.float32 # JAX performs better with float32
        self.int_dtype = jnp.int32
        self.sparse_support = False    # JAX does not support sparse matrices
        self.gpu_support = True        # possible if a specific package is installed (jaxlib with CUDA support)
        self.linalg = jax_linalg
        self.model_generator = JAXModelGenerator()

    def config(self, device=None, float_dtype=None, *args, **kwargs):
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

    @property
    def int_dtype(self):
        return self._int_dtype

    @int_dtype.setter
    def int_dtype(self, value):
        import jax
        self._int_dtype = value
        if value == self.lib.int64 or value == "int64":
            warnings.warn(
                "Using int64 with JAX may lead to slower performance. "
                "Consider using int32 for better performance."
            )
            jax.config.update("jax_enable_x64", True)

    @staticmethod
    @_jax_jit
    def select_values(arr, inds):
        return arr[inds]

    @staticmethod
    @_jax_jit
    def set_values(arr, inds, values):
        arr = arr.at[inds].set(values)
        return arr

    @staticmethod
    @_jax_jit
    def set_flat_values(arr, inds, values):
        arr = arr.at[inds].set(values)
        return arr
    
    @staticmethod
    @_jax_jit
    def add_flat_values(arr, inds, values):
        arr = arr.at[inds].add(values)
        return arr
    
    def copy(self, arr):
        return self.lib.array(arr, dtype=self.float_dtype, copy=True)

    def wrap_sparse(self, csr_matrix, indexes=None, row_reduced=False, local_indexing=False):
        """Converts a sparse matrix in CSR format to JAX compatible ELLPACK format.

        Parameters
        ----------
        csr_matrix : scipy.sparse.csr_matrix
            The input sparse matrix in CSR format.
        indexes : 1D array of int, optional
            Array of indexes where the solution is defined.
            If not provided, all elements are used.
            This parameter is ignored.
        reduced : bool, optional
            Whether to use reduced indexing.
        reindexed : bool, optional
            Whether to reindex the matrix.

        Returns
        -------
        indices : mx.ndarray
            The column indices of the non-zero elements in ELLPACK format.
        data : mx.ndarray
            The non-zero values of the matrix in ELLPACK format.
        """
        if row_reduced and local_indexing:
            raise ValueError("Cannot use both `row_reduced` and `local_indexing` options simultaneously.")
        
        if row_reduced and indexes is None:
            raise ValueError("Indexes must be provided for reduction.")

        if local_indexing and indexes is None:
            raise ValueError("Indexes must be provided for reindexing.")
        
        if local_indexing:
            csr_matrix = csr_matrix[indexes, :][:, indexes]

        ellpack_indices, ellpack_data = csr_to_ellpack(csr_matrix)

        if row_reduced:
            ellpack_indices = ellpack_indices[indexes]
            ellpack_data = ellpack_data[indexes]

        ellpack_indices = self.wrap_indexes(ellpack_indices)
        ellpack_data = self.wrap_array(ellpack_data)

        return ellpack_indices, ellpack_data
