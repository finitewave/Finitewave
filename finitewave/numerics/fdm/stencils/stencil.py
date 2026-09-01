from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp


class Stencil(ABC):
    """
    Base class for diffusion stencils.
    """

    def compute_system_matrices(self, mesh, diffusion, dr, indexes, tissue_indexes=None):
        """
        Computes the weights as a sparse matrix.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        dr : float
            The grid spacing.
        indexes : numpy.ndarray
            The indexes of the non-empty nodes in the mesh.

        Returns
        -------
        scipy.sparse.csr_matrix
            The stiffness matrix.
        scipy.sparse.csr_matrix
            The mass matrix. Diagonal matrix with ones on the diagonal.
        """
        rows, cols, weights = self.compute_diffusion_weights(mesh, diffusion, dr, indexes)
        weights = weights.astype(diffusion.dtype)

        size = mesh.size

        if tissue_indexes is not None:
            rows, cols = self.reindex_matrix(mesh, rows, cols, tissue_indexes)
            size = len(tissue_indexes)

        shape = (size, size)
        # make stiffness matrix with positive diagonal
        K_stiff = sp.csr_matrix((weights, (rows, cols)), shape=shape)
        M_mass = sp.diags(np.ones(K_stiff.shape[0], dtype=weights.dtype),
                          offsets=0, format='csr')
        return K_stiff.tocsr(), M_mass.tocsr()
    
    @abstractmethod
    def compute_diffusion_weights(self, mesh, diffusion, dr, indexes):
        """
        Computes the weights for calculating the diffusion operator.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        dr : float
            The grid spacing.
        indexes : numpy.ndarray
            The indexes of the non-empty nodes in the mesh.

        Returns
        -------
        rows : numpy.ndarray
            The central node indexes.
        cols : numpy.ndarray
            The node indexes involved in diffusion calculation.
        weights : numpy.ndarray
            The weights for connections between central and neighboring nodes.
        """
        raise NotImplementedError()

    def nonzero_weights(self, mesh, ijk, ijk_list, w_list, direction=1):
        """
        Collects non-zero weights.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        ijk : numpy.ndarray
            The indexes of the non-empty nodes in the mesh.
        ijk_list : list
            The list of ijk coordinates of the neighboring nodes in the mesh.
        w_list : list
            The list of weights for the neighboring nodes.
        direction : int, optional
            The direction of the flux (1 for major-to-center, -1 for center-to-major)

        Returns
        -------
        rows : list
            The list of row indexes for the sparse matrix.
        cols : list
            The list of column indexes for the sparse matrix.
        weights : list
            The list of weights for the sparse matrix.
        """
        rows, cols, weights = [], [], []

        for i in range(len(w_list)):
            w = w_list[i]
            ijk_i = ijk_list[i]
            rows += [np.ravel_multi_index(ijk[:, w != 0], mesh.shape)]
            cols += [np.ravel_multi_index(ijk_i[:, w != 0], mesh.shape)]
            weights += [direction * w[w != 0]]

        return rows, cols, weights
             
    def build_neighbor(self, ijk, shift, axis):
        """
        Builds the neighbor ijk coordinates by shifting along a given axis.

        Parameters
        ----------
        ijk : numpy.ndarray
            The ijk coordinates of the current cell.
        shift : int
            The shift to apply along the specified axis.
        axis : int
            The axis along which to apply the shift.

        Returns
        -------
        numpy.ndarray
            The ijk coordinates of the neighbor.
        """

        ijk = ijk.copy()
        ijk[axis] += shift
        return ijk

    def is_valid_index(self, index, mesh):
        """
        Checks if the index coordinates are valid (within bounds and
        in a non-empty cell).

        Parameters
        ----------
        index : numpy.ndarray
            The ijk coordinates of the nodes.
        mesh : numpy.ndarray
            The mesh of the simulation.

        Returns
        -------
        numpy.ndarray
            A boolean array indicating the validity of each index.
        """
        mask = np.ones(index.shape[1], dtype=bool)

        for i in range(mesh.ndim):
            mask &= ((index[i] >= 0) & (index[i] < mesh.shape[i]))

        mask[mask] = mesh[tuple(index[:, mask])] == 1
        return mask
    
    def reindex_matrix(self, mesh, rows, cols, indexes):
        """
        Reindexes the rows and columns of the sparse matrix to avoid zero
        rows in the sparse matrix.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        rows : numpy.ndarray
            The row indices of the sparse matrix.
        cols : numpy.ndarray
            The column indices of the sparse matrix.
        indexes : numpy.ndarray
            The indexes of the non-empty cells in the mesh.

        Returns
        -------
        numpy.ndarray
            The reindexed row indices.
        numpy.ndarray
            The reindexed column indices.
        """
        c_indexes = np.zeros(mesh.size, dtype=np.int64)
        c_indexes[indexes] = np.arange(len(indexes))
        rows = c_indexes[rows]
        cols = c_indexes[cols]
        return rows, cols
