from abc import ABC, abstractmethod
import numpy as np


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
    
    def crs_to_numpy(self, crs_matrix):
        """Converts a sparse matrix in CRS format to NumPy arrays.

        Parameters
        ----------
        crs_matrix : scipy.sparse.csr_matrix
            The input sparse matrix in CRS format.
        
        Returns
        -------
        indptr : np.ndarray
            The index pointer array of the CRS format.
        indices : np.ndarray
            The column indices of the non-zero elements in CRS format.
        data : np.ndarray
            The non-zero values of the matrix in CRS format.
        """
        return crs_matrix.indptr, crs_matrix.indices, crs_matrix.data
    
    def csr_to_ellpack(self, csr_matrix):
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
        ellpack_data = np.zeros((M, K), dtype=np.float64)

        inds = np.repeat([np.arange(K)], M, axis=0)
        mask = inds < row_lengths[:, None]
        ellpack_indices[mask] = csr_matrix.indices
        ellpack_data[mask] = csr_matrix.data.astype(np.float64)

        ellpack_indices = self.simulation.backend.wrap_indexes(ellpack_indices)
        ellpack_data = self.simulation.backend.wrap(ellpack_data)
        
        return ellpack_indices, ellpack_data
