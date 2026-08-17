import numpy as np
from ...core.tissue.cardiac_tissue_base import CardiacTissueBase
from finitewave.core.numerics.fem.elements.element_type import ElementType


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
    def __init__(self, coords, elems, elem_type):
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
        if not ElementType.is_valid(elem_type):
            raise ValueError(f"Invalid element type: {elem_type}.")

        self.meta["shape"] = elem_type
        self.meta["type"] = "Elements"
        self.meta["dim"] = coords.shape[1]
        self.coords = coords
        self.elems = elems
        self.mesh = np.ones(self.coords.shape[0])
        self.mesh_elems = np.ones(self.elems.shape[0])
        self.conductivity = 1.
        self.fibers = None

    @property
    def myo_indexes(self):
        """
        Returns
        -------
        numpy.ndarray
            The flat indices of the ``mesh`` where value is ``1``.
        """
        return np.unique(self.myo_elements.flatten())
    
    @property
    def myo_elements(self):
        """
        Returns
        -------
        numpy.ndarray
            The elements where all nodes are myocytes.
        """
        # TODO: Remove element islands
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
            The flat indices of the ``myo_elements``.
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
    
    def clean(self):
        self.mesh_elems[self.mesh_elems == 2] = 1
        self.mesh[self.mesh == 2] = 1
