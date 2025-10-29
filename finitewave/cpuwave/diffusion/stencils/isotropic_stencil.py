import numpy as np


class IsotropicStencil:
    """
    Assembles the isotropic stencil with first-order boundary conditions.
    """
    def __init__(self):
        pass

    def compute_weights(self, mesh, diffusion, indexes):
        """
        Computes the weights for the isotropic stencil with first-order
        boundary conditions.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        indexes : numpy.ndarray
            The indexes of the non-empty cells in the mesh.

        Returns
        -------
        tuple of np.ndarray
            The rows, columns, and weights for the sparse matrix.
        """
        ijk = np.array(np.unravel_index(indexes, mesh.shape))
        rows = []
        cols = []
        weights = []

        for i_axis in range(mesh.ndim):
            for shift in [-1, 1]:
                diffusion_along_axis = diffusion[*ijk, i_axis, i_axis]
                res = self.compute_flow_weights(diffusion_along_axis, i_axis,
                                                shift, mesh, indexes)
                rows += res[0]
                cols += res[1]
                weights += res[2]

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)
        return rows, cols, weights

    def is_valid_neighbor(self, neighbors, mesh, i_axis):
        """
        Checks if the given neighbors are valid within the mesh.

        Parameters
        ----------
        neighbors : np.ndarray
            The coordinates of the neighboring cells.
        mesh : np.ndarray
            The mesh of the simulation.
        i_axis : int
            The axis index.

        Returns
        -------
        np.ndarray
            A boolean mask indicating valid neighbors.
        """
        mask = ((neighbors[i_axis] >= 0) &
                (neighbors[i_axis] < mesh.shape[i_axis]))
        mask[mask] = mesh[tuple(neighbors[:, mask])] == 1
        return mask.astype(np.int64)

    def compute_flow_weights(self, diffusion, i_axis, shift, mesh, indexes):
        """
        Computes the flow weights for the neighbor in the given axis and shift.

        Parameters
        ----------
        diffusion : np.ndarray
            The diffusion coefficients along the specified axis.
        i_axis : int
            The axis index.
        shift : int
            The shift direction (-1 or 1).
        mesh : np.ndarray
            The mesh of the simulation.
        indexes : np.ndarray
            The indexes of the non-empty cells in the mesh.

        Returns
        -------
        tuple of np.ndarray
            The rows, columns, and weights for the flow computation.
        """
        d = 0.5 * (diffusion + np.roll(diffusion, -shift))
        ijk_center = np.array(np.unravel_index(indexes, mesh.shape))
        ijk_neighbor = ijk_center.copy()
        ijk_neighbor[i_axis] += shift
        valid = self.is_valid_neighbor(ijk_neighbor, mesh, i_axis)
        rows = np.ravel_multi_index(ijk_center[:, valid > 0], mesh.shape)
        cols = np.ravel_multi_index(ijk_neighbor[:, valid > 0], mesh.shape)
        weights = d[valid > 0]
        return [rows, rows], [cols, rows], [weights, -weights]
