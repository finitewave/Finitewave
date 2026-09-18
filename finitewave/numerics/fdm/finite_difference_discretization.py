import scipy.sparse as sp

from finitewave.core.numerics.spatial_discretization import SpatialDiscretization
from .asymmetric_diffusion import AsymmetricDiffusion
from .finite_difference_gradient import FiniteDifferenceGradient


class FiniteDifferenceDiscretization(SpatialDiscretization):
    """Coordinate grid stiffness, identity mass, and gradient operators.

    Attributes
    ----------
    diffusion : AsymmetricDiffusion or IsotropicDiffusion
        Stencil assembler, initially AsymmetricDiffusion(). Assign an
        IsotropicDiffusion instance to use reflected Neumann boundaries.
    gradient : FiniteDifferenceGradient
        Centered and one-sided gradient assembler.

    Notes
    -----
    Matrices use compact C-order indexing of mesh > 0 nodes. Operator
    indexes are flat indices into the full grid, defaulting to mesh == 1.
    K represents negative diffusion in du/dt = -K u; M is identity.

    Examples
    --------
    >>> from finitewave.numerics.fdm.isotropic_diffusion import IsotropicDiffusion
    >>> discretization = FiniteDifferenceDiscretization()
    >>> discretization.diffusion = IsotropicDiffusion()
    """

    def __init__(self):
        self.diffusion = AsymmetricDiffusion()
        self.gradient = FiniteDifferenceGradient()

    def compute_weights(self, tissue, D_model=1.):
        """Build model-scaled stiffness and identity mass from grid tissue.

        Parameters
        ----------
        tissue : CardiacTissueGrid
            Supplies mesh, spacing, compact diffusion tensors, connectivity,
            and the mapping from active nodes to full-grid indices.
        D_model : float, optional
            Multiplier applied to stiffness only, default 1.

        Returns
        -------
        stiffness : scipy.sparse.csr_matrix
            Model-scaled stiffness over all nonempty tissue nodes.
        mass : scipy.sparse.csr_matrix
            Identity matrix of matching shape and dtype.
        """
        indexes = tissue.tissue_indexes[tissue.myo_indexes]
        K, M = self.build_system_matrices(
            tissue.mesh, tissue.dr, indexes, tissue.diffusion_tensor,
            tissue.connectivity)
        return K * D_model, M

    def build_system_matrices(self, mesh, dr=1., indexes=None,
                              diffusion=1., connectivity=1.):
        """Build stiffness and identity mass in compact tissue indexing.

        Parameters
        ----------
        mesh : numpy.ndarray
            Grid labels: 0 for empty, 1 for active, 2 for fibrotic nodes.
            All mesh > 0 nodes are included in compact C-order matrix indexing.
        dr : float, optional
            Uniform grid spacing along every axis. Default is 1.0.
        indexes : numpy.ndarray, optional
            Flat full-grid indices of center nodes. For operator builders,
            None selects mesh == 1; these are not compact matrix indices. Default is None.
        diffusion : float or numpy.ndarray, optional
            Isotropic coefficient or tensors of shape (N_tissue, ndim, ndim),
            ordered by flat indices where mesh > 0. N_tissue counts nonempty nodes. Default is 1.0.
        connectivity : float or numpy.ndarray, optional
            Connection factors: scalar, one per axis, or an array with shape
            (N_tissue, ndim), passed to the stencil connectivity helper. Default is 1.0.

        Returns
        -------
        stiffness : scipy.sparse.csr_matrix
            Matrix of shape (N_tissue, N_tissue), representing negative diffusion
            in du/dt = -K u. It includes the 1/dr**2 scaling.
        mass : scipy.sparse.csr_matrix
            Identity matrix with the same shape and dtype as stiffness.
        """
        K = self.build_diffusion_operator(mesh, dr, indexes, diffusion, connectivity)
        M = sp.eye(K.shape[0], dtype=K.dtype, format='csr')
        return K, M

    def build_diffusion_operator(self, mesh, dr=1., indexes=None,
                                 diffusion=1., connectivity=1.):
        """Build the selected diffusion stiffness matrix.

        Parameters
        ----------
        mesh : numpy.ndarray
            Grid labels: 0 for empty, 1 for active, 2 for fibrotic nodes.
            All mesh > 0 nodes are included in compact C-order matrix indexing.
        dr : float, optional
            Uniform grid spacing along every axis. Default is 1.0.
        indexes : numpy.ndarray, optional
            Flat full-grid indices of center nodes. For operator builders,
            None selects mesh == 1; these are not compact matrix indices. Default is None.
        diffusion : float or numpy.ndarray, optional
            Isotropic coefficient or tensors of shape (N_tissue, ndim, ndim),
            ordered by flat indices where mesh > 0. N_tissue counts nonempty nodes. Default is 1.0.
        connectivity : float or numpy.ndarray, optional
            Connection factors: scalar, one per axis, or an array with shape
            (N_tissue, ndim), passed to the stencil connectivity helper. Default is 1.0.

        Returns
        -------
        stiffness : scipy.sparse.csr_matrix
            Matrix of shape (N_tissue, N_tissue), representing negative diffusion
            in du/dt = -K u. It includes the 1/dr**2 scaling.
        """
        return self.diffusion.build_diffusion_operator(
            mesh, dr, indexes, diffusion, connectivity)

    def build_gradient_operator(self, mesh, *, dr=1., indexes=None, **kwargs):
        """Build centered or one-sided grid gradient operators.

        Parameters
        ----------
        mesh : numpy.ndarray
            Grid labels: 0 for empty, 1 for active, 2 for fibrotic nodes.
            All mesh > 0 nodes are included in compact C-order matrix indexing.
        dr : float, optional
            Uniform grid spacing along every axis. Default is 1.0.
        indexes : numpy.ndarray, optional
            Flat full-grid indices of center nodes. For operator builders,
            None selects mesh == 1; these are not compact matrix indices. Default is None.
        **kwargs : dict
            Ignored; accepted for compatibility with other discretizations.

        Returns
        -------
        grad_ops : list of scipy.sparse.csr_matrix
            One matrix per grid axis, each of shape (N_tissue, N_tissue).
            Apply to values ordered by flat indices where mesh > 0. Uses centered
            differences with two active neighbors, one-sided differences with one,
            and zero with neither. Rows outside indexes remain zero.
        """
        return self.gradient.build_gradient_operator(mesh, dr=dr, indexes=indexes, **kwargs)
