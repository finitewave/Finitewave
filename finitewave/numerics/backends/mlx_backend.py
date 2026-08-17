import warnings
import numpy as np
from finitewave.core.backend.backend import Backend


class MlxBackend(Backend):
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

    def wrap_sparse(self, csr_matrix, indexes, local_indexing=False):
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
        if local_indexing:
            csr_matrix = csr_matrix[indexes, :][:, indexes]

        rows_len = np.diff(csr_matrix.indptr)
        n_cols = np.max(rows_len)
        n_rows = csr_matrix.shape[0]

        ellpack_indices = np.repeat(np.arange(n_rows), n_cols).reshape(n_rows, n_cols)
        ellpack_data = np.zeros((n_rows, n_cols), dtype=np.float32)

        inds = np.repeat([np.arange(n_cols)], n_rows, axis=0)
        mask = inds < rows_len[:, None]
        ellpack_indices[mask] = csr_matrix.indices
        ellpack_data[mask] = csr_matrix.data.astype(np.float32)
        # A@x = x, otherwise x=0 for empty rows, which is not correct
        ellpack_indices[n_rows == 0, 0] = 1. 

        ellpack_indices = self.wrap_indexes(ellpack_indices)
        ellpack_data = self.wrap_array(ellpack_data)
        indexes = self.wrap_indexes(indexes)

        return ellpack_indices, ellpack_data, indexes