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

    def __init__(self, shape=None, dr=None, mesh=None):
        """
        Initializes the CardiacTissue on a regular grid.

        Parameters
        ----------
        shape : tuple, optional
            The shape of the tissue grid. Required if mesh is not supplied.
            If both are supplied, this must match mesh.shape.
        dr : float
            The spatial resolution of the grid
            (distance between adjacent points).
        mesh : np.ndarray, optional
            A 2D or 3D numpy array representing the tissue mesh where each value
            indicates the type of tissue at that location. Stored without copying.
        """
        super().__init__()
        if dr is None:
            raise TypeError("dr is required for grid tissue.")
        if shape is None and mesh is None:
            raise TypeError("Supply shape or mesh for grid tissue.")
        if shape is not None and mesh is not None:
            if tuple(shape) != mesh.shape:
                raise ValueError("Shape of the mesh does not match the provided shape.")

        if mesh is not None:
            self.mesh = mesh
        elif shape is not None:
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

        d_ac, d_al = self._prepare_diffusion_coefficients()

        outer_product = np.einsum('ij,ik->ijk', fibers, fibers, optimize='optimal')
        diffusion_tensor = (d_ac[..., None, None] * np.eye(ndim) + 
                            (d_al - d_ac)[..., None, None] * outer_product)
        return diffusion_tensor * np.atleast_1d(conductivity)[:, None, None]

    def _prepare_diffusion_coefficients(self):
        """
        Prepares the diffusion coefficients for the tissue.

        Returns
        -------
            A tuple containing the diffusion coefficients for the tissue.
        """
        D_ac = self.D_ac
        D_al = self.D_al

        if np.asarray(D_ac).shape == self.mesh.shape:
            D_ac = D_ac[self.mesh > 0]

        if np.asarray(D_al).shape == self.mesh.shape:
            D_al = D_al[self.mesh > 0]

        return np.atleast_1d(D_ac), np.atleast_1d(D_al)

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
