import numpy as np
from .isotropic_stencil import IsotropicStencil


class IsotropicSecondOrderStencil(IsotropicStencil):
    """
    Assembles the isotropic stencil with second-order boundary conditions.
    """

    def __init__(self):
        super().__init__()

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

        ijk_opposite = ijk_center.copy()
        ijk_opposite[i_axis] -= shift

        valid_neighbor = self.is_valid_neighbor(ijk_neighbor, mesh, i_axis)
        valid_opposite = self.is_valid_neighbor(ijk_opposite, mesh, i_axis)

        rows = np.ravel_multi_index(ijk_center[:, valid_neighbor > 0],
                                    mesh.shape)
        cols = np.ravel_multi_index(ijk_neighbor[:, valid_neighbor > 0],
                                    mesh.shape)

        weights = d * valid_neighbor * (valid_neighbor + valid_opposite > 0)
        weights = weights[valid_neighbor > 0]
        return [rows, rows], [cols, rows], [weights, -weights]
