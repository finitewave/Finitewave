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
        self.mesh = np.ones(shape, dtype=np.int8)
        self.D_ac = 1. / 9.
        self.D_al = 1
        self.conductivity = 1.
        self.connectivity = 1.
        self.local_index_map = None
        self.dr = dr
        self.fibers = None

    @property
    def meta(self):
        """
        Returns
        -------
        dict
            A dictionary containing metadata about the tissue.
        """
        return {
            "dim": self.mesh.ndim,
            "shape": self.mesh.shape,
            "type": "Grid",
            "dr": self.dr
        }

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

    @property
    def tissue_index_map(self):
        """
        Returns
        -------
        numpy.ndarray
            A mapping from the flat indices of the ``mesh`` to the flat indices
            of the tissue mesh (where mesh value is not ``0``).
        """
        tissue_index_map = - np.zeros_like(self.mesh, dtype=np.int64)
        tissue_indexes = self.tissue_indexes
        tissue_index_map.flat[tissue_indexes] = np.arange(len(tissue_indexes))
        return tissue_index_map

    @property
    def diffusion_tensor(self):
        """
        Returns
        -------
        numpy.ndarray
            The diffusion tensor of the tissue.
        """
        fibers = self._extract_tissue_fibers()

        if fibers is None and np.asarray(self.conductivity).size == 1:
            return self.conductivity

        conductivity = self.conductivity
        if np.asarray(conductivity).shape == self.mesh.shape:
            conductivity = conductivity[self.mesh > 0]

        ndim = self.meta["dim"]

        if fibers is None:
            tissue_shape = self.mesh[self.mesh > 0].shape
            diffusion_tensor = np.zeros(tissue_shape + (ndim, ndim))
            for i in range(ndim):
                diffusion_tensor[..., i, i] = conductivity
            return diffusion_tensor

        outer_product = np.einsum('ij,ik->ijk', fibers, fibers, optimize='optimal')
        diffusion_tensor = self.D_ac * np.eye(ndim) + (self.D_al - self.D_ac) * outer_product
        return diffusion_tensor * np.atleast_1d(conductivity)[:, None, None]

    def _extract_tissue_fibers(self):
        if self.fibers is None:
            return None

        if self.fibers.shape == self.mesh.shape + (self.mesh.ndim,):
            return self.fibers[self.mesh > 0]

        if self.fibers.shape == self.mesh[self.mesh > 0].shape + (self.mesh.ndim,):
            return self.fibers

        raise ValueError(
            f"Fibers shape {self.fibers.shape} is not compatible with mesh" +
            f" shape {self.mesh.shape} or tissue shape {self.mesh[self.mesh > 0].shape}."
        )
