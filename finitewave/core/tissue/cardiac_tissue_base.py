from abc import ABC, abstractmethod
import copy
import numpy as np


class CardiacTissueBase(ABC):
    """Base class for a model tissue.

    This class represents the tissue model used in cardiac simulations.
    It includes attributes and methods for defining the tissue structure,
    ts properties, and handling fibrosis patterns.

    Attributes
    ----------
    meta : dict
        A dictionary containing metadata about the tissue.
    """
    def __init__(self):
        self.meta = {}
        self.D_ac = 1/9
        self.D_al = 1

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
