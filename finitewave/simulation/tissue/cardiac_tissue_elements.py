import numpy as np
from finitewave.core.tissue.cardiac_tissue_base import CardiacTissueBase
from finitewave.numerics.fem.elements.element_type import ElementType


class CardiacTissueElements(CardiacTissueBase):
    """
    Class representing cardiac tissue defined by elements.

    Attributes
    ----------
    meta : dict
        A dictionary containing metadata about the tissue.
    coords : np.ndarray
        An (N, D) array of coordinates for the N nodes in D-dimensional space.
    elems : np.ndarray
        An (M, E) array of element connectivity, where M is the number of
        elements and E is the number of nodes per element.
    mesh : np.ndarray
        A 1D array of length N indicating the type of tissue at each node.
        Possible values are: ``0`` for non-tissue, ``1`` for healthy tissue, and
        ``2`` for fibrotic tissue.
    mesh_elems : np.ndarray
        A 1D array of length M indicating the type of tissue for each element.
        Possible values are: ``0`` for non-tissue, ``1`` for healthy tissue, and
        ``2`` for fibrotic tissue.
    conductivity : float or np.ndarray
        The conductivity of the elements used for reducing the diffusion coefficients.
        The conductivity should be in the range [0, 1].
    fibers : np.ndarray
        Fibers orientation in the elements. If None, the isotropic stencil is
        used.
    """
    def __init__(self, coords, elems, elem_type, order=1):
        """
        Initializes the CardiacTissue on a finite element mesh.

        Parameters
        ----------
        coords : np.ndarray
            An (N, D) array of coordinates for the N nodes in D-dimensional space.
        elems : np.ndarray
            An (M, E) array of element connectivity, where M is the number of
            elements and E is the number of nodes per element.
        elem_type : str
            A string indicating the type of elements.
        """
        super().__init__()
        self.coords = coords
        self.elems = elems
        self.mesh = np.ones(self.coords.shape[0])
        self.mesh_elems = np.ones(self.elems.shape[0])
        self.conductivity = 1.
        self.fibers = None
        self.reference_element = ElementType.select_reference_element(elem_type, order)

    @property
    def meta(self):
        return {
            "dim": self.coords.shape[1],
            "shape": self.reference_element.name,
            "order": self.reference_element.order,
            "type": "Elements"
        }

    @property
    def myo_coords(self):
        """
        Returns
        -------
        numpy.ndarray
            The coordinates of the myocytes in the tissue.
        """
        return self.coords[self.myo_indexes]

    @property
    def myo_indexes(self):
        """
        Returns
        -------
        numpy.ndarray
            The flat indices of the ``mesh`` where value is ``1``.
        """
        return np.unique(self.myo_elems.flatten())
    
    @property
    def myo_elems(self):
        """
        Returns
        -------
        numpy.ndarray
            The elements where all nodes are myocytes.
        """
        myo_nodes_mask = self.mesh == 1
        myo_elems_mask = ((self.mesh_elems == 1) &
                          np.all(myo_nodes_mask[self.elems], axis=1))
        return self.elems[myo_elems_mask]
    
    @property
    def myo_elems_indexes(self):
        """
        Returns
        -------
        numpy.ndarray
            The flat indices of the ``myo_elems``.
        """
        return np.flatnonzero(self.myo_elems_mask)
    
    @property
    def myo_elems_mask(self):
        """
        Returns
        -------
        numpy.ndarray
            A boolean array of length M indicating which elements are myocytes.
        """
        myo_nodes_mask = self.mesh == 1
        myo_elems_mask = ((self.mesh_elems == 1) & np.all(myo_nodes_mask[self.elems], axis=1))
        return myo_elems_mask
    
    @property
    def tissue_indexes(self):
        """
        Returns
        -------
        numpy.ndarray
            The flat indices of the ``mesh`` where value is greater than ``0``.
        """
        return np.arange(self.mesh.size)

    @property
    def diffusion_tensor(self):
        if self.fibers is None:
            diffusion_tensor = np.zeros((self.elems.shape[0], self.meta["dim"], self.meta["dim"]))
            for i in range(self.meta["dim"]):
                diffusion_tensor[:, i, i] = self.conductivity
            return diffusion_tensor

        outer_product = np.einsum('ij,ik->ijk', self.fibers, self.fibers, optimize='optimal')
        diffusion_tensor = self.D_ac * np.eye(self.meta["dim"]) + (self.D_al - self.D_ac) * outer_product
        return diffusion_tensor * np.atleast_1d(self.conductivity)[:, None, None]
    
    def clean(self):
        self.mesh_elems[self.mesh_elems == 2] = 1
        self.mesh[self.mesh == 2] = 1
