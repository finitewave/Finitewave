import numpy as np

from finitewave.core.tissue.cardiac_tissue import CardiacTissue


class CardiacTissue2D(CardiacTissue):
    """
    This class represents a 2D cardiac tissue.

    Attributes
    ----------
    meta : dict
        A dictionary containing metadata about the tissue.
    mesh : np.ndarray
        A 2D numpy array representing the tissue mesh where each value
        indicates the type of tissue at that location. Possible values are:
        ``0`` for non-tissue, ``1`` for healthy tissue, and ``2`` for fibrotic
        tissue.
    conductivity : float or np.ndarray
        The conductivity of the tissue used for reducing the diffusion
        coefficients. The conductivity should be in the range [0, 1].
    fibers : np.ndarray
        Fibers orientation in the tissue. If None, the isotropic stencil is
        used.
    """

    def __init__(self, shape):
        super().__init__()
        self.meta["dim"] = 2
        self.meta["shape"] = shape
        self.mesh = np.ones(shape, dtype=np.int8)
        self.conductivity = 1.0
        self.fibers = None

    def add_boundaries(self):
        """
        Sets the boundary values of the mesh to zero.

        The boundaries are defined as the edges of the grid, and this method
        updates these edges in the mesh array.
        """
        self._mesh[0, :] = 0
        self._mesh[:, 0] = 0
        self._mesh[-1, :] = 0
        self._mesh[:, -1] = 0
