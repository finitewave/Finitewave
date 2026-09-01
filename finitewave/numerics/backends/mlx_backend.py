from .linalg.sparse import csr_to_ellpack

from finitewave.core.backend.backend import Backend


def _check_mlx_installed():
    try:
        import mlx.core
    except ImportError as exc:
        if exc.name == "mlx" or exc.name.startswith("mlx."):
            raise ImportError(
                "MlxBackend requires the optional 'mlx' package."
            ) from exc
        raise


class MlxBackend(Backend):
    def __init__(self):
        _check_mlx_installed()
        import mlx.core as mx
        from .linalg import mlx_linalg
        from .model.mlx_model_generator import MLXModelGenerator

        self.name = "mlx"
        self.lib = mx
        self.float_dtype = mx.float32  # MLX GPU supports only float32 (float64 is only for CPU)
        self.int_dtype = mx.int32
        self.sparse_support = False    # MLX does not support sparse matrices
        self.gpu_support = True
        self.linalg = mlx_linalg
        self.model_generator = MLXModelGenerator()

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

    def sync_backend(self, *args):
        self.lib.eval(args)

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

    def wrap_sparse(self, csr_matrix, indexes=None, row_reduced=False, local_indexing=False):
        """Converts a sparse matrix in CSR format to JAX compatible ELLPACK format.

        Parameters
        ----------
        csr_matrix : scipy.sparse.csr_matrix
            The input sparse matrix in CSR format.
        indexes : 1D array of int, optional
            Array of indexes where the solution is defined.
            this parameter is ignored.
        local_indexing : bool, optional
            Whether to use local indexing.

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
        
