from abc import ABC, abstractmethod


class FibrosisPattern(ABC):
    """Abstract base class for generating and applying fibrosis patterns to
    cardiac tissue.

    This class defines an interface for creating fibrosis patterns and applying
    them to cardiac tissue models. Subclasses must implement the ``generate``
    method to define specific patterns. The ``apply`` method uses the generated
    pattern to modify the mesh of the cardiac tissue.
    """
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, shape=None, mesh=None):
        """
        Generates a fibrosis pattern for the given shape and optionally based
        on the provided mesh.

        Parameters
        ----------
        shape : tuple
            The shape of the mesh (e.g., (ni, nj) or (ni, nj, nk)).
        mesh : numpy.ndarray, optional
            The existing mesh to base the pattern on. Default is None.

        Returns
        -------
        numpy.ndarray
            A new mesh array with the applied fibrosis pattern.
        """

    def apply(self, cardiac_tissue):
        """
        Applies the generated fibrosis pattern to the specified cardiac tissue
        object.

        This method calls the ``generate`` method to create the pattern and
        then updates the ``mesh`` attribute of the ``cardiac_tissue`` object
        with the generated pattern.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue
            The cardiac tissue object to which the fibrosis pattern will be
            applied. The ``mesh`` attribute of this object will be updated with
            the generated pattern.
        """
        cardiac_tissue.mesh = self.generate(mesh=cardiac_tissue.mesh)
