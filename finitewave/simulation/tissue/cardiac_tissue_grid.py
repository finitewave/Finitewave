import numpy as np
from finitewave.core.tissue.cardiac_tissue_base import CardiacTissueBase



class CardiacTissueGrid(CardiacTissueBase):
    """
    Class representing a cardiac tissue on a regular grid.

    Attributes
    ----------
    meta : dict
        A dictionary containing metadata about the tissue.
    mesh : np.ndarray
        A 2D or 3D numpy array representing the tissue mesh where each value
        indicates the type of tissue at that location. Possible values are:
        ``0`` for non-tissue, ``1`` for healthy tissue, and ``2`` for fibrotic
        tissue.
    conductivity : float or np.ndarray (mesh.shape)
        The conductivity of the nodes in the tissue used for reducing the diffusion
        coefficients. The conductivity should be in the range [0, 1].
    connectivity : float or np.ndarray (mesh.shape + (ndim,))
        The conductivity of the junctions between nodes in the tissue.
        The connectivity between node (i, ...) and (i+1, ...) should be given in
        connectivity[i, ..., 0] etc.
    fibers : np.ndarray
        Fibers orientation in the tissue. If None, the isotropic stencil is
        used.
    """

    def __init__(self, shape, dr):
        """
        Initializes the CardiacTissue on a regular grid.

        Parameters
        ----------
        shape : tuple
            The shape of the tissue grid.
        dr : float
            The spatial resolution of the grid
            (distance between adjacent points).
        """
        super().__init__()
        self.meta["dim"] = len(shape)
        self.meta["shape"] = shape
        self.meta["type"] = "Grid"
        self.meta["dr"] = dr
        self.mesh = np.ones(shape, dtype=np.int8)
        self.D_ac = 1. / 9.
        self.D_al = 1
        self.conductivity = 1.
        self.connectivity = 1.
        self.dr = dr
        self.fibers = None

    @property
    def coords(self):
        """
        Returns
        -------
        numpy.ndarray
            The coordinates of all points in the tissue mesh.
        """
        return np.argwhere(self.mesh >= 0)

    @property
    def myo_coords(self):
        """
        Returns
        -------
        numpy.ndarray
            The coordinates of the myocytes in the tissue mesh.
        """
        return np.argwhere(self.mesh == 1)

    @property
    def tissue_coords(self):
        """
        Returns
        -------
        numpy.ndarray
            The coordinates of the tissue in the tissue mesh.
        """
        return np.argwhere(self.mesh > 0)

    @property
    def myo_indexes(self):
        """
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
        Returns
        -------
        numpy.ndarray
            The flat indices of the ``mesh`` where value is not ``0``.
        """
        return np.flatnonzero(self.mesh > 0)
