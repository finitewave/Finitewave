import numpy as np

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class Diffuse3DPattern(FibrosisPattern):
    """
    A class to generate a diffuse fibrosis pattern in a 3D mesh grid.

    Attributes
    ----------
    x1, x2 : int
        The start and end indices for the region of interest along the x-axis.
    y1, y2 : int
        The start and end indices for the region of interest along the y-axis.
    z1, z2 : int
        The start and end indices for the region of interest along the z-axis.
    dens : float
        The density of fibrosis within the specified region, ranging from 0 (no fibrosis) to 1 (full fibrosis).

    Methods
    -------
    generate(size, mesh=None):
        Generates a 3D mesh with a diffuse fibrosis pattern within the specified region.
    """

    def __init__(self, x1, x2, y1, y2, z1, z2, density):
        """
        Initializes the Diffuse3DPattern object with the given region of interest and density.

        Parameters
        ----------
        x1, x2 : int
            The start and end indices for the region of interest along the x-axis.
        y1, y2 : int
            The start and end indices for the region of interest along the y-axis.
        z1, z2 : int
            The start and end indices for the region of interest along the z-axis.
        dendensitys : float
            The density of fibrosis within the specified region.
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2
        self.density = density

    def generate(self, shape, mesh=None):
        """
        Generates a 3D mesh with a diffuse fibrosis pattern within the specified region.

        If a mesh is provided, the pattern is applied to the existing mesh; otherwise, a new mesh is created.

        Parameters
        ----------
        shape : tuple of int
            The size of the 3D mesh grid (x, y, z).
        mesh : numpy.ndarray, optional
            A 3D NumPy array representing the existing mesh grid to which the fibrosis pattern will be applied.
            If None, a new mesh grid of the given size is created.

        Returns
        -------
        numpy.ndarray
            A 3D NumPy array of the same shape as the input, with the diffuse fibrosis pattern applied.
        """

        if shape is None and mesh is None:
            message = "Either shape or mesh must be provided."
            raise ValueError(message)

        if shape is not None:
            mesh = np.ones(shape, dtype=np.int8)
            fibr = self._generate(mesh.shape)
            mesh[self.x1: self.x2, self.y1: self.y2, self.z1: self.z2] = fibr[self.x1: self.x2,
                                                                              self.y1: self.y2,
                                                                              self.z1: self.z2]
            return mesh

        fibr = self._generate(mesh.shape)
        mesh[self.x1: self.x2, self.y1: self.y2, self.z1, self.z2] = fibr[self.x1: self.x2,
                                                                          self.y1: self.y2,
                                                                          self.z1: self.z2]
        return mesh

    def _generate(self, shape):
        return 1 + (np.random.random(shape) <= self.density).astype(np.int8)
