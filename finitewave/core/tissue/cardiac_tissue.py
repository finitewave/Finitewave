from abc import ABC, abstractmethod
import copy
import numpy as np


class CardiacTissue(ABC):
    """Base class for a model tissue.

    This class represents the tissue model used in cardiac simulations.
    It includes attributes and methods for defining the tissue structure,
    ts properties, and handling fibrosis patterns.

    Attributes
    ----------
    meta : dict
        A dictionary containing metadata about the tissue.
    special_boundaries : np.ndarray
        An array containing labels for special boundaries in the tissue mesh.
        This array is used to define Dirichlet boundary conditions as points
        with non-zero values are ignored in the solver.
    """
    def __init__(self):
        self.meta = {}
        self.special_boundaries = None

    @property
    def mesh(self):
        """
        Gets the tissue mesh array.

        Returns
        -------
        numpy.ndarray
            The tissue mesh array.
        """
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        """
        Sets the tissue mesh array.

        Parameters
        ----------
        mesh : numpy.ndarray
            The tissue mesh array.
        """
        if mesh.ndim != self.meta['dim']:
            raise ValueError("Mesh dimension must match the tissue dimension.")

        self._mesh = mesh
        self.add_boundaries()

    def compute_myo_indexes(self):
        """
        Computes flat indices of the myocytes in the tissue mesh.
        """
        if self.special_boundaries is not None:
            self.myo_indexes = np.flatnonzero((self.mesh == 1) &
                                              (self.special_boundaries == 0))
            return

        self.myo_indexes = np.flatnonzero(self.mesh == 1)

    @abstractmethod
    def add_boundaries(self):
        """
        Abstract method to be implemented by subclasses for adding boundary
        conditions to the tissue mesh.
        """
        pass

    def add_pattern(self, fibro_pattern):
        """
        Applies a fibrosis pattern to the tissue mesh.

        Parameters
        ----------
        fibro_pattern : FibrosisPattern
            A fibrosis pattern object to apply to the tissue mesh.
        """
        fibro_pattern.apply(self)

    def clean(self):
        """
        Removes all fibrosis points from the mesh, setting them to ``1``
        (healthy tissue).
        """
        self.mesh[self.mesh == 2] = 1

    def clone(self):
        """
        Creates a deep copy of the current ``CardiacTissue`` instance.

        Returns
        -------
        CardiacTissue
            A deep copy of the current ``CardiacTissue`` instance.
        """
        return copy.deepcopy(self)
