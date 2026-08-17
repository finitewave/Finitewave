import warnings

from finitewave.core.backend.backend import Backend


class NumbaBackend(Backend):
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
        import numba

        if device not in (None, "cpu"):
            raise ValueError("Numba backend supports only device='cpu'.")

        if num_of_threads is None:
            return

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

    def wrap_matrix(self, crs_matrix, indexes, local_indexing=False):
        """Converts a sparse matrix in CRS format to NumPy arrays.
        The output arrays are filtered to only include the rows and columns
        specified by the indexes.

        Parameters
        ----------
        crs_matrix : scipy.sparse.csr_matrix
            The input sparse matrix in CRS format.
        dtype : np.dtype
            The data type for the output arrays.
        indexes : np.ndarray
            Array of indexes where the solution is defined.

        Returns
        -------
        indptr : np.ndarray
            The index pointer array of the CRS format.
        indices : np.ndarray
            The column indices of the non-zero elements in CRS format.
        data : np.ndarray
            The non-zero values of the matrix in CRS format.
        """
        if local_indexing:
            crs_matrix = crs_matrix[:, indexes][indexes, :]

        data = self.wrap_array(crs_matrix.data)
        indptr = self.wrap_indexes(crs_matrix.indptr)
        indices = self.wrap_indexes(crs_matrix.indices)
        indexes = self.wrap_indexes(indexes)
        return indptr, indices, data, indexes
