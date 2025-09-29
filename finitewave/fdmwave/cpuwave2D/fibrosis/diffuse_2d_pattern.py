import numpy as np

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class Diffuse2DPattern(FibrosisPattern):
    """
    Class for generating a diffuse 2D fibrosis pattern in a given mesh area.

    Attributes
    ----------
    density : float
        The density of the fibrosis in the specified area
    x1 : int
        The starting x-coordinate of the fibrosis area.
    x2 : int
        The ending x-coordinate of the fibrosis area.
    y1 : int
        The starting y-coordinate of the fibrosis area.
    y2 : int
        The ending y-coordinate of the fibrosis area.
    """

    def __init__(self, density, x1=None, x2=None, y1=None, y2=None):
        """
        Initializes the Diffuse2DPattern with the specified parameters.

        Parameters
        ----------
        density : float
            The density of the fibrosis in the specified area.
        x1 : int
            The starting x-coordinate of the fibrosis area.
        x2 : int
            The ending x-coordinate of the fibrosis area.
        y1 : int
            The starting y-coordinate of the fibrosis area.
        y2 : int
            The ending y-coordinate of the fibrosis area.
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.density = density

    def generate(self, shape=None, mesh=None):
        """
        Generates a diffuse 2D fibrosis pattern for the given shape and mesh.
        The resulting pattern is applied to the mesh within the specified
        area.

        Parameters
        ----------
        shape : tuple
            The shape of the mesh.
        mesh : numpy.ndarray, optional
            The existing mesh to base the pattern on. Default is None.

        Returns
        -------
        numpy.ndarray
            A new mesh array with the applied fibrosis pattern.

        Notes
        -----
        If both parameters are provided, first non-None parameter is used.
        """

        if shape is None and mesh is None:
            message = "Either shape or mesh must be provided."
            raise ValueError(message)

        if shape is not None:
            mesh = np.ones(shape, dtype=np.int8)
            fibr = self._generate(mesh.shape)
            mesh[self.x1: self.x2, self.y1: self.y2] = fibr[self.x1: self.x2,
                                                            self.y1: self.y2]
            return mesh

        fibr = self._generate(mesh.shape)
        mesh[self.x1: self.x2, self.y1: self.y2] = fibr[self.x1: self.x2,
                                                        self.y1: self.y2]
        return mesh

    def _generate(self, shape):
        return 1 + (np.random.random(shape) <= self.density).astype(np.int8)
