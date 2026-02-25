import numpy as np
import scipy.sparse as sp


class Stencil:
    """
    Base class for diffusion stencils.
    """

    def compute_system_matrices(self, mesh, diffusion, dr, indexes, reindex=False):
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
            The indexes of the non-empty points in the mesh.

        Returns
        -------
        scipy.sparse.csr_matrix
            The stiffness matrix.
        scipy.sparse.csr_matrix
            The mass matrix. Diagonal matrix with ones on the diagonal.
        """
        rows, cols, weights = self.compute_diffusion_weights(mesh, diffusion, dr, indexes)
        weights = weights.astype(diffusion.dtype)

        if reindex:
            rows, cols = self.reindex_matrix(mesh, rows, cols, indexes)

        size = len(indexes)
        shape = (size, size)
        # make stiffness matrix with positive diagonal
        K_stiff = sp.csr_matrix((weights, (rows, cols)), shape=shape)
        M_mass = sp.diags(np.ones_like(indexes, dtype=weights.dtype),
                          offsets=0, format='csr')
        return K_stiff.tocsr(), M_mass.tocsr()

    def compute_diffusion_weights(self, mesh, diffusion, dr, indexes):
        """
        Computes the weights for the isotropic stencil with first-order
        boundary conditions.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        dr : float
            The grid spacing.
        indexes : numpy.ndarray
            The indexes of the non-empty points in the mesh.

        Returns
        -------
        tuple of np.ndarray
            The rows, columns, and weights for the sparse matrix.
        """
        rows = []
        cols = []
        weights = []

        ijk = np.array(np.unravel_index(indexes, mesh.shape))
        for axis in range(mesh.ndim):
            res = self.compute_flux_weights(mesh, diffusion, dr, ijk, axis)
            ijk_major, ijk_list, w_list = res
            r, c, w = self.compute_diffusion_component(mesh, dr, ijk, ijk_major, (ijk_list, w_list))
            rows.append(r)
            cols.append(c)
            weights.append(w)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)
        return rows, cols, weights
    
    def compute_diffusion_component(self, mesh, dr, ijk, ijk_major, flux_weights):
        """
        Computes the diffusion weights from the flux weights.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        dr : float
            The grid spacing.
        ijk : numpy.ndarray
            The indexes of the non-empty points in the mesh.
        flux_weights : list
            A list of tuples containing the rows, cols, and weights for each
            flux direction.

        Returns
        -------
        rows : np.ndarray
            The row indexes for the sparse matrix.
        cols : np.ndarray
            The column indexes for the sparse matrix.
        weights : np.ndarray
            The weights for the sparse matrix.
        """
        ijk_list, w_list = flux_weights
        in_flux = self.nonzero_weights(mesh, ijk, ijk_list, w_list)
        out_flux = self.nonzero_weights(mesh, ijk_major, ijk_list, w_list, direction=-1)

        rows = np.concatenate(in_flux[0] + out_flux[0])
        cols = np.concatenate(in_flux[1] + out_flux[1])
        weights = np.concatenate(in_flux[2] + out_flux[2]) / dr
        return rows, cols, weights

    def compute_flux_weights(self, mesh, diffusion, dr, ijk, axis):
        """
        Computes the flux weights along the given axis.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        dr : float
            The grid spacing.
        ijk : numpy.ndarray
            The indexes of the non-empty points in the mesh.
        axis : int
            The axis along which to compute the flux weights.

        Returns
        -------
        tuple of np.ndarray
            The rows, columns, and weights for the flux computation.
        """
        pass

    def nonzero_weights(self, mesh, ijk, ijk_list, w_list, direction=1):
        """
        Collects non-zero weights.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        ijk : numpy.ndarray
            The indexes of the non-empty points in the mesh.
        ijk_list : list
            The list of ijk coordinates of the involved points in the mesh.
        w_list : list
            The list of weights for the involved points.
        direction : int, optional
            The direction of the flux (1 for major-to-center, -1 for center-to-major)

        Returns
        -------
        tuple
            A tuple containing the rows, cols, and weights for the non-zero
            weights.

        Notes
        -----
        This method applies to both major-to-center and center-to-major flows.
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
            The ijk coordinates of the points.
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
