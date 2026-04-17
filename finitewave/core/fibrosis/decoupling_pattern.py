import numpy as np


class DecouplingPattern:
    """
    Class for computing the decoupling pattern for fibrotic tissue.

    The decoupling pattern is a binary mask that indicates which nodes in the
    tissue are decoupled (i.e., have zero diffusion) due to fibrosis. This
    pattern is used to modify the diffusion coefficients in the simulation.

    Attributes
    ----------
    coords : np.ndarray
        An array of coordinates where the decoupling pattern should be applied.
    axis : int or None
        The direction along which to apply the decoupling pattern.
    density : float or None
        The density of randomly decoupled nodes in the tissue.
    region : list of tuples or None
        The region within which to apply the random decoupling pattern,
        specified as a list of tuples (min, max) for each dimension.
    """

    def __init__(self, coords=None, axis=None, density=None, region=None):
        """Initializes the DecouplingPattern with the specified parameters.
        
        Parameters
        ----------
        coords : np.ndarray (N, ndim), optional
            An array of coordinates where the decoupling pattern should be applied.
        axis : int or None, optional
            The direction along which to apply the decoupling pattern.
            If None, the pattern will be applied randomly across all axes.
        density : float or None, optional
            The density of randomly decoupled nodes in the tissue.
        region : list of tuples or None, optional
            The region within which to apply the random decoupling pattern,
            specified as a list of tuples (min, max) for each dimension.
            If None and density is provided, the random decoupling will be
            applied across the entire tissue.
        """
        self.coords = coords
        self.axis = axis
        self.density = density
        self.region = region

    def apply(self, cardiac_tissue):
        if np.isscalar(cardiac_tissue.connectivity):
            cond = np.ones(cardiac_tissue.mesh.shape + (cardiac_tissue.mesh.ndim,), dtype=np.float64)
            cardiac_tissue.connectivity *= cond

        self._line_decoupling(cardiac_tissue)
        self._random_decoupling(cardiac_tissue)
        return cardiac_tissue

    def _line_decoupling(self, cardiac_tissue):
        """
        Builds a line in the decoupling pattern between the specified coordinates.

        Parameters
        ----------
        cardiac_tissue : CardiacTissueGrid
            The cardiac tissue to which the decoupling pattern will be applied.

        Returns
        -------
        CardiacTissueGrid
            The modified cardiac tissue with the applied decoupling pattern.
        """
        if self.coords is None or self.axis is None:
            return cardiac_tissue

        cardiac_tissue.connectivity[*tuple(self.coords.T), self.axis] = 0.0
        return cardiac_tissue
    
    def _random_decoupling(self, cardiac_tissue):
        """
        Randomly decouples nodes in the cardiac tissue based on the specified density.

        Parameters
        ----------
        cardiac_tissue : CardiacTissueGrid
            The cardiac tissue to which the random decoupling will be applied.

        Returns
        -------
        CardiacTissueGrid
            The modified cardiac tissue with the applied random decoupling pattern.
        """
        if self.density is None:
            return cardiac_tissue
        
        coords = np.array(np.where(cardiac_tissue.mesh == 1))
        coords = coords[:, np.random.rand(coords.shape[1]) < self.density]
        
        if self.region is not None:
            for dim in range(coords.shape[0]):
                coords = coords[:, (coords[dim] >= self.region[dim][0]) & (coords[dim] < self.region[dim][1])]
        
        axis = self.axis
        if axis is None:
            axis = np.random.randint(0, coords.shape[0], size=coords.shape[1])

        cardiac_tissue.connectivity[*tuple(coords), axis] = 0.0

        return cardiac_tissue
