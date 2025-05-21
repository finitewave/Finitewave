import numpy as np
import random

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class Structural3DPattern(FibrosisPattern):
    """
    A class to generate a structural fibrosis pattern in a 3D mesh grid.

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
    length_i, length_j, length_k : int
        The lengths of fibrosis blocks along each axis (x, y, z).

    Methods
    -------
    generate(size, mesh=None):
        Generates a 3D mesh with a structural fibrosis pattern within the specified region.
    """

    def __init__(self, x1, x2, y1, y2, z1, z2, density, length_i, length_j, length_k):
        """
        Initializes the Structural3DPattern object with the given region of interest, density, and block sizes.

        Parameters
        ----------
        x1, x2 : int
            The start and end indices for the region of interest along the x-axis.
        y1, y2 : int
            The start and end indices for the region of interest along the y-axis.
        z1, z2 : int
            The start and end indices for the region of interest along the z-axis.
        density : float
            The density of fibrosis within the specified region.
        length_i, length_j, length_k : int
            The lengths of fibrosis blocks along each axis (x, y, z).
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2
        self.density = density
        self.length_i = length_i
        self.length_j = length_j
        self.length_k = length_k

    def generate(self, shape=None, mesh=None):
        """
        Generates and applies a structural fibrosis pattern to the mesh.

        The mesh is divided into blocks of size `length_i` x `length_j` x `length_k`, with each block having 
        a probability `density` of being filled with fibrosis. The function ensures that blocks do not
        extend beyond the specified region.

        Parameters
        ----------
        shape : tuple
            The shape of the mesh.
        mesh : numpy.ndarray, optional
            The existing mesh to base the pattern on. Default is None..

        Returns
        -------
        numpy.ndarray
            A new mesh array with the applied fibrosis pattern.
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
        mesh[self.x1: self.x2, self.y1: self.y2, self.z1: self.z2] = fibr[self.x1: self.x2,
                                                                          self.y1: self.y2,
                                                                          self.z1: self.z2]
        return mesh

    def _generate(self, shape, mesh=None):
        """
        Generates a 3D mesh with a structural fibrosis pattern within the specified region.

        If a mesh is provided, the pattern is applied to the existing mesh; otherwise, a new mesh is created.

        Parameters
        ----------
        shape : tuple of int
            The shape of the 3D mesh grid (x, y, z).
        mesh : numpy.ndarray, optional
            A 3D NumPy array representing the existing mesh grid to which the fibrosis pattern will be applied.
            If None, a new mesh grid of the given size is created.

        Returns
        -------
        numpy.ndarray
            A 3D NumPy array of the same shape as the input, with the structural fibrosis pattern applied.
        """
        mesh = np.ones(shape, dtype=np.int8)
        for i in range(self.x1, self.x2, self.length_i):
            for j in range(self.y1, self.y2, self.length_j):
                for k in range(self.z1, self.z2, self.length_k):
                    if random.random() <= self.density:
                        i_s = min(self.length_i, self.x2 - i)
                        j_s = min(self.length_j, self.y2 - j)
                        k_s = min(self.length_k, self.z2 - k)

                        mesh[i:i+i_s, j:j+j_s, k:k+k_s] = 2

        return mesh
