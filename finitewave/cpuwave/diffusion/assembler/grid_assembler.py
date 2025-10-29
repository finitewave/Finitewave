import numpy as np
from scipy import sparse as sp


class GridAssembler:
    """
    This class computes the weights for solver on 2D and 3D grids.

    Attributes
    ----------
    stencil : Stencil
        An instance of the Stencil class to compute the weights for the
        diffusion operator.
    """

    def __init__(self):
        self.stencil = None

    def assemble_matrices(self, simulation):
        """
        Computes the weights for isotropic diffusion in 2D.

        Parameters
        ----------
        simulation : simulation
            A simulation object containing the simulation parameters.

        Returns
        -------
        scipy.sparse.csr_matrix
            The stiffness matrix for the asymmetric stencil.
        scipy.sparse.csr_matrix
            The mass matrix for the asymmetric stencil.
        """
        tissue = simulation.cardiac_tissue
        model = simulation.cardiac_model

        mesh = tissue.mesh.copy()
        mesh[mesh != 1] = 0

        diffusion = self.compute_diffusion(mesh, tissue.conductivity,
                                           tissue.fibers, tissue.D_al,
                                           tissue.D_ac, model.D_model,
                                           simulation.dr)
        diffusion = diffusion.astype(simulation.npfloat)

        stiff, mass = self.compute_weights_sparse(mesh, diffusion,
                                                  tissue.myo_indexes)
        return stiff, mass

    def compute_weights_sparse(self, mesh, diffusion, indexes):
        """
        Computes the weights for the asymmetric stencil as a sparse matrix.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        indexes : numpy.ndarray
            The indexes of the non-empty cells in the mesh.

        Returns
        -------
        scipy.sparse.csr_matrix
            The stiffness matrix for the asymmetric stencil.
        scipy.sparse.csr_matrix
            The mass matrix for the asymmetric stencil.
        """
        rows, cols, weights = self.stencil.compute_weights(mesh, diffusion,
                                                           indexes)
        weights = weights.astype(diffusion.dtype)
        rows, cols = self.reindex_matrix(mesh, rows, cols, indexes)

        size = len(indexes)
        shape = (size, size)
        # make stiffness matrix with positive diagonal
        K_stiff = - sp.csr_matrix((weights, (rows, cols)), shape=shape)
        M_mass = sp.diags(np.ones_like(indexes, dtype=weights.dtype),
                          offsets=0, format='csr')
        return K_stiff.tocsr(), M_mass.tocsr()

    def reindex_matrix(self, mesh, rows, cols, indexes):
        """
        Reindexes the rows and columns of the sparse matrix to avoid zero
        rows in the sparse matrix.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        rows : numpy.ndarray
            The row indices of the sparse matrix.
        cols : numpy.ndarray
            The column indices of the sparse matrix.
        indexes : numpy.ndarray
            The indexes of the non-empty cells in the mesh.

        Returns
        -------
        numpy.ndarray
            The reindexed row indices.
        numpy.ndarray
            The reindexed column indices.
        """
        c_indexes = np.zeros(mesh.size, dtype=np.int64)
        c_indexes[indexes] = np.arange(len(indexes))
        rows = c_indexes[rows]
        cols = c_indexes[cols]
        return rows, cols

    def compute_diffusion(self, mesh, conductivity, fibers, D_al, D_ac,
                          D_model, dr):
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
            diffusion = np.zeros(mesh.shape + (ndim, ndim), dtype=mesh.dtype)
            for i in range(ndim):
                diffusion[..., i, i] = conductivity * D_model / (dr ** 2)
            return diffusion

        ndim = fibers.shape[-1]
        diffusion = np.zeros(mesh.shape + (ndim, ndim), dtype=fibers.dtype)
        for i in range(ndim):
            for j in range(ndim):
                d_ij = self.compute_diffusion_components(fibers, i, j, D_al, D_ac)
                diffusion[..., i, j] = d_ij * conductivity * D_model / (dr ** 2)
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
