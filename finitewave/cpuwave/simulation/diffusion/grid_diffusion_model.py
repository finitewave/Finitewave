import numpy as np
from finitewave.core.diffusion.diffusion_model_base import DiffusionModelBase
from finitewave.cpuwave.numerics.fdm.asymmetric_stencil import AsymmetricStencil


class GridDiffusionModel(DiffusionModelBase):
    """
    Diffusion model for grid-based simulations.

    This model uses a finite difference stencil to compute the diffusion
    weights, which are then used in the time integration step of the
    simulation.

    Attributes
    ----------
    stencil : Stencil
        The stencil used for computing diffusion weights.
    simulation : CardiacSimulation
        The simulation instance associated with this diffusion model.
    diffusion : np.ndarray
        The diffusion tensor        
    weights : tuple
        Stiffness and mass matrices
    """
    def __init__(self):
        super().__init__()
        self.stencil = AsymmetricStencil()

    def initialize(self, simulation):
        """
        Computes the weights for the diffusion operator on a grid.
        """
        self.simulation = simulation
        self.compute_weights()

    def compute_weights(self):
        """
        Computes the weights for the diffusion operator on a grid.

        Returns
        -------
        scipy.sparse.csr_matrix
            The stiffness matrix for the asymmetric stencil.
        scipy.sparse.csr_matrix
            The mass matrix for the asymmetric stencil.
        """
        tissue = self.simulation.cardiac_tissue
        model = self.simulation.cardiac_model

        mesh = tissue.mesh.copy()
        mesh[mesh != 1] = 0

        diffusion = self.compute_diffusion(mesh, tissue.conductivity,
                                           tissue.fibers, tissue.D_al,
                                           tissue.D_ac, model.D_model)
        self.diffusion = diffusion.astype(self.simulation.npfloat)

        self.weights = self.stencil.compute_system_matrices(mesh, self.diffusion,
                                                            tissue.dr, tissue.myo_indexes,
                                                            reindex=True)
        return self.weights

    def compute_diffusion(self, mesh, conductivity, fibers, D_al, D_ac, D_model):
        """
        Computes the diffusion tensor based on fiber orientations.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        conductivity : numpy.ndarray
            The conductivity values for each cell in the mesh.
        fibers : np.ndarray
            Array representing fiber orientations.
        D_al : float
            Longitudinal diffusion coefficient.
        D_ac : float
            Cross-sectional diffusion coefficient.
        D_model : float
            Model specific diffusion coefficient.

        Returns
        -------
        np.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        """
        # Isotropic case when fibers are not provided
        if fibers is None:
            ndim = mesh.ndim
            diffusion = np.zeros(mesh.shape + (ndim, ndim), dtype=np.float64)
            for i in range(ndim):
                diffusion[..., i, i] = conductivity * D_model
            return diffusion

        ndim = fibers.shape[-1]
        diffusion = np.zeros(mesh.shape + (ndim, ndim), dtype=np.float64)
        for i in range(ndim):
            for j in range(ndim):
                d_ij = self.compute_diffusion_components(fibers, i, j, D_al, D_ac)
                diffusion[..., i, j] = d_ij * conductivity * D_model
        return diffusion

    def compute_diffusion_components(self, fibers, i_axis, j_axis, D_al, D_ac):
        """
        Computes the diffusion components based on fiber orientations.

        Parameters
        ----------
        fibers : np.ndarray
            Array representing fiber orientations.
        i_axis : int
            First axis index.
        j_axis : int
            Second axis index.
        D_al : float
            Longitudinal diffusion coefficient.
        D_ac : float
            Cross-sectional diffusion coefficient.

        Returns
        -------
        np.ndarray
            Array of diffusion components based on fiber orientations
        """
        return (D_ac * (i_axis == j_axis) +
                (D_al - D_ac) * fibers[..., i_axis] * fibers[..., j_axis])