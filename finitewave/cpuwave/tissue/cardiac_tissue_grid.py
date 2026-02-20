import numpy as np

from finitewave.core.tissue.cardiac_tissue import CardiacTissue


class CardiacTissueGrid(CardiacTissue):
    """
    This class represents a cardiac tissue.

    Attributes
    ----------
    meta : dict
        A dictionary containing metadata about the tissue.
    mesh : np.ndarray
        A 2D or 3D numpy array representing the tissue mesh where each value
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

    def __init__(self, shape, dr=1.0):
        super().__init__()
        self.meta["dim"] = len(shape)
        self.meta["shape"] = shape
        self.meta["type"] = "Grid"
        self.meta["dr"] = dr
        self.mesh = np.ones(shape, dtype=np.int8)
        self.conductivity = 1.0
        self.dr = dr
        self.fibers = None

    @property
    def coords(self):
        """
        Gets the coordinates of all points in the tissue mesh.

        Returns
        -------
        numpy.ndarray
            The coordinates of all points in the tissue mesh.
        """
        return np.argwhere(self.mesh >= 0)

    @property
    def myo_coords(self):
        """
        Gets the coordinates of the myocytes in the tissue mesh.

        Returns
        -------
        numpy.ndarray
            The coordinates of the myocytes in the tissue mesh.
        """
        return np.argwhere(self.mesh == 1)

    @property
    def tissue_coords(self):
        """
        Gets the coordinates of the tissue in the tissue mesh.

        Returns
        -------
        numpy.ndarray
            The coordinates of the tissue in the tissue mesh.
        """
        return np.argwhere(self.mesh > 0)

    @property
    def myo_indexes(self):
        """
        Gets the flat indices of the myocytes in the tissue mesh.

        Returns
        -------
        numpy.ndarray
            The flat indices of the ``mesh`` where value is ``1``.
        """
        return np.flatnonzero(self.mesh == 1)

    @property
    def myo_on_tissue_indexes(self):
        """
        Gets the flat indices of the myocytes on the ``tissue_indexes``.

        Returns
        -------
        numpy.ndarray
            The indices of the ``tissue_indexes`` where mesh value is ``1``.
        """
        mesh = self.mesh[self.mesh > 0]
        return np.flatnonzero(mesh == 1)

    @property
    def tissue_indexes(self):
        """
        Gets the flat indices of the tissue in the tissue mesh.

        Returns
        -------
        numpy.ndarray
            The flat indices of the ``mesh`` where value is not ``0``.
        """
        return np.flatnonzero(self.mesh > 0)
