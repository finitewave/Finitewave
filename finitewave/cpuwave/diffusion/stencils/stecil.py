import numpy as np


class Stencil:
    """
    Base class for diffusion stencils.
    """

    def compute_weights(self, mesh, diffusion, indexes):
        """
        Computes the weights for the stencil.

        Parameters
        ----------
        mesh : np.ndarray
            The mesh of the simulation.
        diffusion : np.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        indexes : np.ndarray
            The indexes of the non-empty cells in the mesh.

        Returns
        -------
        tuple
            A tuple containing:
            - rows: np.ndarray
                The row indexes for the sparse matrix.
            - cols: np.ndarray
                The column indexes for the sparse matrix.
            - weights: np.ndarray
                The weights for the sparse matrix.
        """
        raise NotImplementedError

    def nonzero_weights(self, mesh, ijk_center, ijk_major, ijk_list, w_list):
        """
        Collects non-zero weights and their corresponding rows and columns.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        ijk_center : numpy.ndarray
            The indexes of the center cells.
        ijk_major : numpy.ndarray
            The indexes of the major neighbor cells.
        ijk_list : list
            The list of ijk coordinates of the involved cells.
        w_list : list
            The list of weights for the involved cells.

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
            rows += [np.ravel_multi_index(ijk_center[:, w != 0], mesh.shape)]
            cols += [np.ravel_multi_index(ijk_i[:, w != 0], mesh.shape)]
            weights += [-w[w != 0]]

            rows += [np.ravel_multi_index(ijk_major[:, w != 0], mesh.shape)]
            cols += [np.ravel_multi_index(ijk_i[:, w != 0], mesh.shape)]
            weights += [w[w != 0]]

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

    def is_valid_neighbor(self, neighbor, mesh):
        """
        Checks if the neighbor ijk coordinates are valid (within bounds and
        in a non-empty cell).

        Parameters
        ----------
        neighbor : numpy.ndarray
            The ijk coordinates of the neighbor cell.
        mesh : numpy.ndarray
            The mesh of the simulation.

        Returns
        -------
        numpy.ndarray
            A boolean array indicating the validity of each neighbor.
        """
        mask = np.ones(neighbor.shape[1], dtype=bool)

        for i in range(mesh.ndim):
            mask &= ((neighbor[i] >= 0) & (neighbor[i] < mesh.shape[i]))

        mask[mask] = mesh[tuple(neighbor[:, mask])] == 1
        return mask.astype(mesh.dtype)
