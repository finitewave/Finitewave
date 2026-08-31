import numpy as np
from finitewave.core.diffusion.diffusion_model_base import DiffusionModelBase
from finitewave.numerics.fdm.stencils.asymmetric_stencil import AsymmetricStencil


class DiffusionModel(DiffusionModelBase):
    """
    Class for assembling grid-based diffusion operator.

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
        self.update_weights()

    def update_weights(self):
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
        return self.compute_weights(tissue, model.D_model)
    
    def compute_weights(self, tissue, D_model=1.0):
        """
        Computes the weights for the diffusion operator on a grid.

        Parameters
        ----------
        tissue : CardiacTissue
            The cardiac tissue for which to compute the diffusion weights.
        D_model : float, optional
            Model-specific diffusion coefficient. Default is 1.0.

        Returns
        -------
        scipy.sparse.csr_matrix
            The stiffness matrix for the asymmetric stencil.
        scipy.sparse.csr_matrix
            The mass matrix for the asymmetric stencil.
        """
        mesh = tissue.mesh.copy()
        mesh[mesh != 1] = 0 

        self.diffusion = self.compute_diffusion_tensor(
            mesh, tissue.conductivity, tissue.fibers, tissue.D_al, tissue.D_ac, D_model
        )
        myo_indexes = tissue.tissue_indexes[tissue.myo_indexes]
        self.connectivity_tensor = self.convert_to_connectivity(self.diffusion, tissue.connectivity)
        self.weights = self.stencil.compute_system_matrices(
            mesh, self.connectivity_tensor, tissue.dr, myo_indexes, reindex=True
        )
        return self.weights

    def compute_diffusion_tensor(self, mesh, conductivity, fibers, D_al, D_ac, D_model):
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
            
            diffusion[mesh == 0, ...] = 0
            return diffusion

        ndim = fibers.shape[-1]
        diffusion = np.zeros(mesh.shape + (ndim, ndim), dtype=np.float64)
        for i in range(ndim):
            for j in range(ndim):
                d_ij = self.compute_diffusion_components(fibers, i, j, D_al, D_ac)
                diffusion[..., i, j] = d_ij * conductivity * D_model

        diffusion[mesh == 0, ...] = 0
        return diffusion
    
    def convert_to_connectivity(self, diffusion, connectivity):
        """
        Computes the diffusion tensor for junctions between myocytes (nodes).

        Parameters
        ----------
        diffusion : np.ndarray (mesh.shape + (ndim, ndim))
            The diffusion tensor for the myocytes.
        connectivity : float or np.ndarray (mesh.shape + (ndim,))
            The conductivity coefficient for the junctions between myocytes.

        Returns
        -------
        np.ndarray (mesh.shape + (ndim, ndim))
            The diffusion tensor for the junctions between myocytes.
        """
        connectivity_tensor = np.zeros_like(diffusion)

        dim = diffusion.shape[-2]
        
        for d in range(dim):

            if diffusion.shape[d] < 2:
                continue
            
            index_left = [slice(None)] * (dim + 2)
            index_left[d] = slice(None, -1)
            index_left[dim] = d
            index_left = tuple(index_left)
            
            index_right = [slice(None)] * (dim + 2)
            index_right[d] = slice(1, None)
            index_right[dim] = d
            index_right = tuple(index_right)

            d_i = (diffusion[index_left] + diffusion[index_right]) / 2.0
            connectivity_tensor[index_left] = d_i
        
        connectivity_tensor[diffusion == 0] = 0

        if np.isscalar(connectivity):
            connectivity_tensor *= connectivity
        else:
            connectivity_tensor *= connectivity[..., None]

        return connectivity_tensor

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
        return (D_ac * (i_axis == j_axis) + (D_al - D_ac) * fibers[..., i_axis] * fibers[..., j_axis])
