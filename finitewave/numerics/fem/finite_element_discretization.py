import numpy as np
import scipy.sparse as sp

from finitewave.core.numerics.spatial_discretization import SpatialDiscretization
from .finite_element_diffusion import FiniteElementDiffusion
from .finite_element_gradient import FiniteElementGradient


class FiniteElementDiscretization(SpatialDiscretization):
    """Coordinate finite-element stiffness, mass, and gradient assembly.

    Parameters
    ----------
    reference_element : object, optional
        Reference shape functions and quadrature data. Supply this before
        building mesh operators, or call ``compute_weights`` to use the
        tissue's reference element.

    Attributes
    ----------
    diffusion : FiniteElementDiffusion
        Assembles stiffness and provides quadrature geometry helpers.
    gradient : FiniteElementGradient
        Builds center-gradient operators.
    reference_element : object
        Assign through this property to update both operators together.

    Notes
    -----
    Mass assembly belongs to this class. ``build_system_matrices`` reuses
    one set of quadrature Jacobians for stiffness and mass. Triangle,
    tetrahedron, quadrilateral, and hexahedron integration uses 3, 4, 4,
    and 8 points respectively. Distorted tensor-product elements are
    integrated numerically; stiffness need not be integrated exactly.
    """
    def __init__(self, reference_element=None):
        self._reference_element = reference_element
        self.diffusion = FiniteElementDiffusion(reference_element)
        self.gradient = FiniteElementGradient(reference_element)

    @property
    def reference_element(self):
        """Reference element; assignment synchronizes diffusion and gradient operators.
        """
        return self._reference_element

    @reference_element.setter
    def reference_element(self, value):
        self._reference_element = value
        self.diffusion.reference_element = value
        self.gradient.reference_element = value

    def compute_weights(self, tissue, D_model=1.):
        """
        Computes the weights for the diffusion operator.

        Parameters
        ----------
        tissue : CardiacTissueBase
            The tissue object containing the mesh and diffusion tensor.
            N_nodes is the number of rows in ``tissue.coords``; nodes outside
            ``tissue.myo_elems`` retain zero rows and columns in the matrices.
        D_model : float, optional
            The diffusion coefficient to scale the stiffness matrix, by default 1.

        Returns
        -------
        sparse.csr_matrix
            The stiffness matrix with shape (N_nodes, N_nodes).
        sparse.csr_matrix
            The mass matrix with shape (N_nodes, N_nodes).
        """
        diffusion = tissue.diffusion_tensor
        coords = tissue.coords
        elems = tissue.myo_elems
        self.reference_element = tissue.reference_element
        K, M = self.build_system_matrices(coords, elems, diffusion)
        return K * D_model, M

    def build_system_matrices(self, coords, elems, diffusion=1.):
        """Build stiffness and consistent mass with shared quadrature Jacobians.

        Parameters
        ----------
        coords : numpy.ndarray, shape (N_nodes, dim_phys)
            Physical coordinates of all mesh nodes, including unused nodes.
        elems : numpy.ndarray, shape (N_elems, N_points)
            Integer connectivity using indices into ``coords``.
        diffusion : float or numpy.ndarray, optional
            Isotropic coefficient or one physical tensor per element with shape
            (N_elems, dim_phys, dim_phys). Constant within each element. Default is 1.

        Returns
        -------
        stiffness : scipy.sparse.csr_matrix
            Stiffness matrix of shape (N_nodes, N_nodes), integrating
            ``grad(N_i).T @ diffusion @ grad(N_j)``. Unused nodes retain zero
            rows and columns. This is K in ``M du/dt = -K u`` for pure diffusion
            with zero-flux boundaries, not the strong operator ``div(D grad(u))``.
        mass : scipy.sparse.csr_matrix
            Consistent mass matrix of shape (N_nodes, N_nodes).
        """
        jacobian = self.diffusion._build_integration_jacobian(coords, elems)
        stiffness = self.diffusion._build_diffusion_operator(coords, elems, diffusion, jacobian)
        mass = self._build_mass_matrix(coords, elems, jacobian)
        return stiffness, mass

    def build_diffusion_operator(self, coords, elems, diffusion=1.):
        """Delegate stiffness assembly to ``FiniteElementDiffusion``.

        Parameters
        ----------
        coords : numpy.ndarray, shape (N_nodes, dim_phys)
            Physical coordinates of all mesh nodes, including unused nodes.
        elems : numpy.ndarray, shape (N_elems, N_points)
            Integer connectivity using indices into ``coords``.
        diffusion : float or numpy.ndarray, optional
            Isotropic coefficient or one physical tensor per element with shape
            (N_elems, dim_phys, dim_phys). Constant within each element. Default is 1.

        Returns
        -------
        stiffness : scipy.sparse.csr_matrix
            Stiffness matrix of shape (N_nodes, N_nodes), integrating
            ``grad(N_i).T @ diffusion @ grad(N_j)``. Unused nodes retain zero
            rows and columns. This is K in ``M du/dt = -K u`` for pure diffusion
            with zero-flux boundaries, not the strong operator ``div(D grad(u))``.
        """
        return self.diffusion.build_diffusion_operator(coords, elems, diffusion)

    def build_gradient_operator(self, coords, elems, *, as_sparse=True, **kwargs):
        """Build physical gradients evaluated at element centers.

        Parameters
        ----------
        coords : numpy.ndarray, shape (N_nodes, dim_phys)
            Physical coordinates of all mesh nodes, including unused nodes.
        elems : numpy.ndarray, shape (N_elems, N_points)
            Integer connectivity using indices into ``coords``.
        as_sparse : bool, optional
            Return one CSR matrix per physical axis when True (default).
        **kwargs : dict
            Ignored; accepted for compatibility with other discretizations.

        Returns
        -------
        grads : tuple of scipy.sparse.csr_matrix or numpy.ndarray
            Sparse matrices have shape (N_elems, N_nodes) and map nodal values
            to element gradients. With ``as_sparse=False``, returns shape-function
            gradients of shape (N_elems, dim_phys, N_points). Gradients are
            constant within linear triangles/tetrahedra and evaluated at the
            reference center for quadrilaterals/hexahedra. Embedded surfaces
            return tangential gradients in physical coordinates.
        """
        return self.gradient.build_gradient_operator(
            coords, elems, as_sparse=as_sparse, **kwargs)

    def build_mass_matrix(self, coords, elems):
        """Build the consistent mass matrix by integrating shape-function products.

        Parameters
        ----------
        coords : numpy.ndarray, shape (N_nodes, dim_phys)
            Physical coordinates of all mesh nodes, including unused nodes.
        elems : numpy.ndarray, shape (N_elems, N_points)
            Integer connectivity using indices into ``coords``.

        Returns
        -------
        mass : scipy.sparse.csr_matrix
            Consistent mass matrix of shape (N_nodes, N_nodes). Unused nodes
            retain zero rows and columns.
        """
        jacobian = self.diffusion._build_integration_jacobian(coords, elems)
        return self._build_mass_matrix(coords, elems, jacobian)

    def _build_mass_matrix(self, coords, elems, jacobian):
        """Assemble consistent mass using existing quadrature Jacobians.

        Parameters
        ----------
        coords : numpy.ndarray, shape (N_nodes, dim_phys)
            Physical coordinates of all mesh nodes, including unused nodes.
        elems : numpy.ndarray, shape (N_elems, N_points)
            Integer connectivity using indices into ``coords``.
        jacobian : numpy.ndarray, shape (N_elems, N_quad, dim_ref, dim_phys)
            Jacobians at quadrature points for the supplied mesh.

        Returns
        -------
        mass : scipy.sparse.csr_matrix
            Consistent mass matrix of shape (N_nodes, N_nodes). Unused nodes
            retain zero rows and columns.
        """
        rows, cols = self.diffusion._build_matrix_rows_cols(elems)
        weights = self.diffusion._compute_integration_weights(jacobian)
        shape_values = self.reference_element.integration_N
        shape = (coords.shape[0], coords.shape[0])

        mass_data = np.einsum('eq,qi,qj->eij', weights,
                              shape_values, shape_values, optimize=True)
        mass_data = mass_data.flatten()
        mass_matrix = sp.coo_matrix((mass_data, (rows, cols)), shape=shape)
        return mass_matrix.tocsr()
