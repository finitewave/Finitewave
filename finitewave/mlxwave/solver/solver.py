import numpy as np
import mlx.core as mx
from abc import ABC, abstractmethod


class Solver(ABC):
    """
    A base class for solvers used to solve linear systems.
    """
    def __init__(self):
        pass

    @abstractmethod
    def assemble_system(self, stiffness_matrix, mass_matrix, dt):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError
    
    def build_ellpack(self, csr_matrix):
        """Converts a sparse matrix in CSR format to ELLPACK format.

        Parameters
        ----------
        csr_matrix : scipy.sparse.csr_matrix
            The input sparse matrix in CSR format.

        Returns
        -------
        indices : mx.ndarray
            The column indices of the non-zero elements in ELLPACK format.
        data : mx.ndarray
            The non-zero values of the matrix in ELLPACK format.
        """
        row_lengths = np.diff(csr_matrix.indptr)
        K = np.max(row_lengths)
        M = csr_matrix.shape[0]

        ellpack_indices = np.repeat(np.arange(M), K).reshape(M, K)
        ellpack_data = np.zeros((M, K), dtype=np.float32)

        inds = np.repeat([np.arange(K)], M, axis=0)
        mask = inds < row_lengths[:, None]
        ellpack_indices[mask] = csr_matrix.indices
        ellpack_data[mask] = csr_matrix.data.astype(np.float32)

        ellpack_indices = mx.array(ellpack_indices, dtype=mx.int32)
        ellpack_data = mx.array(ellpack_data, dtype=mx.float32)

        return ellpack_indices, ellpack_data
