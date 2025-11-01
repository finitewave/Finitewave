import numpy as np
from .stecil import Stencil


class IsotropicStencil(Stencil):
    """
    Assembles the isotropic stencil with first-order boundary conditions.
    """
    def __init__(self):
        super().__init__()

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

        ijk = np.array(np.unravel_index(indexes, mesh.shape))
        for axis in range(mesh.ndim):
            res = self.compute_flow_weights(mesh, diffusion, ijk, axis)
            rows += res[0]
            cols += res[1]
            weights += res[2]

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)
        return rows, cols, weights

    def compute_flow_weights(self, mesh, diffusion, ijk, axis):
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
        d_axis = diffusion[..., axis, axis]
        ijk_neighbor = self.build_neighbor(ijk, shift=1, axis=axis)
        m_valid = self.is_valid_neighbor(ijk_neighbor, mesh)
        weights = self.average_diffusion(d_axis, ijk, ijk_neighbor, m_valid)

        ijk_list = [ijk, ijk_neighbor]
        w_list = [-weights, weights]

        return self.nonzero_weights(mesh, ijk, ijk_neighbor, ijk_list, w_list)

    def average_diffusion(self, diffusion, ijk, ijk_neighbor, mask):
        """
        Computes the average diffusion between the center and neighbor cells.

        Parameters
        ----------
        diffusion : np.ndarray
            The diffusion coefficients.
        ijk : np.ndarray
            The indices of the center cells.
        ijk_neighbor : np.ndarray
            The indices of the neighbor cells.
        mask : np.ndarray
            The mask indicating valid neighbors.

        Returns
        -------
        np.ndarray
            The average diffusion coefficients.
        """
        d_avg = np.zeros(mask.shape, dtype=diffusion.dtype)
        d_avg[mask > 0] = 0.5 * (diffusion[tuple(ijk[:, mask > 0])] +
                                 diffusion[tuple(ijk_neighbor[:, mask > 0])])
        return d_avg
