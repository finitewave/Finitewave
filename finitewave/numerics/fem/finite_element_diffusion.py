import numpy as np
import scipy.sparse as sp
from finitewave.numerics.fem.finite_element_gradient import invert_jacobian


class FiniteElementDiffusion:
    """Assemble the diffusion stiffness matrix by quadrature.

    All supported elements use the same integration path. Diffusion is scalar
    or a constant physical tensor per element. Distorted tensor-product
    elements use numerical quadrature, which need not integrate stiffness
    exactly.

    Parameters
    ----------
    reference_element : object, optional
        Shape values, derivatives, and weights at the integration points.
    """

    def __init__(self, reference_element=None):
        self.reference_element = reference_element

    def build_diffusion_operator(self, coords, elems, diffusion=1.):
        """Build the diffusion stiffness matrix by quadrature.

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
        jacobian = self._build_integration_jacobian(coords, elems)
        stiffness = self._build_diffusion_operator(coords, elems, diffusion, jacobian)
        return stiffness

    def _build_diffusion_operator(self, coords, elems, diffusion, jacobian):
        """Assemble stiffness using existing quadrature Jacobians.

        Parameters
        ----------
        coords : numpy.ndarray, shape (N_nodes, dim_phys)
            Physical coordinates of all mesh nodes, including unused nodes.
        elems : numpy.ndarray, shape (N_elems, N_points)
            Integer connectivity using indices into ``coords``.
        diffusion : float or numpy.ndarray, optional
            Isotropic coefficient or one physical tensor per element with shape
            (N_elems, dim_phys, dim_phys). Constant within each element. Default is 1.
        jacobian : numpy.ndarray, shape (N_elems, N_quad, dim_ref, dim_phys)
            Jacobians at quadrature points for the supplied mesh.

        Returns
        -------
        stiffness : scipy.sparse.csr_matrix
            Stiffness matrix of shape (N_nodes, N_nodes), integrating
            ``grad(N_i).T @ diffusion @ grad(N_j)``. Unused nodes retain zero
            rows and columns. This is K in ``M du/dt = -K u`` for pure diffusion
            with zero-flux boundaries, not the strong operator ``div(D grad(u))``.
        """
        if np.isscalar(diffusion):
            diffusion = np.repeat(np.eye(coords.shape[1])[None, :, :],
                                    elems.shape[0], axis=0) * diffusion

        shape = (coords.shape[0], coords.shape[0])
        rows, cols = self._build_matrix_rows_cols(elems)
        weights = self._compute_integration_weights(jacobian)
        grads = self._compute_integration_gradients(jacobian)

        stiff_data = np.einsum('eq,eqki,ekl,eqlj->eij', weights,
                               grads, diffusion, grads, optimize=True)
        stiff_data = stiff_data.flatten()
        stiff_matrix = sp.coo_matrix((stiff_data, (rows, cols)), shape=shape)
        return stiff_matrix.tocsr()

    def _build_matrix_rows_cols(self, elems):
        """Build sparse assembly indices for element matrices.

        Parameters
        ----------
        elems : numpy.ndarray, shape (N_elems, N_points)
            Integer element connectivity.

        Returns
        -------
        rows : numpy.ndarray
            Flattened row indices of length N_elems * N_points**2.
        cols : numpy.ndarray
            Matching column indices in the same order as flattened local matrices.
        """
        n_elem_points = elems.shape[1]
        rows = np.repeat(elems, n_elem_points, axis=1).ravel()
        cols = np.tile(elems, (1, n_elem_points)).ravel()
        return rows, cols

    def _compute_integration_gradients(self, jacobian):
        """Transform reference derivatives into physical gradients.

        Parameters
        ----------
        jacobian : numpy.ndarray, shape (N_elems, N_quad, dim_ref, dim_phys)
            Jacobians at quadrature points.

        Returns
        -------
        grads : numpy.ndarray, shape (N_elems, N_quad, dim_phys, N_points)
            Physical shape-function gradients; tangential for embedded surfaces.
        """
        return np.einsum('eqpr,qri->eqpi', invert_jacobian(jacobian),
                         self.reference_element.integration_dN, optimize=True)

    def _compute_integration_weights(self, jacobian):
        """Scale reference quadrature weights by physical area or volume.

        Parameters
        ----------
        jacobian : numpy.ndarray, shape (N_elems, N_quad, dim_ref, dim_phys)
            Jacobians at quadrature points. Supported shapes are (2, 2),
            (2, 3), and (3, 3) for the last two axes.

        Returns
        -------
        weights : numpy.ndarray, shape (N_elems, N_quad)
            Quadrature weights multiplied by the absolute determinant, or by
            the cross-product norm for surfaces in 3D. Summing the quadrature
            axis gives the numerically integrated area or volume per element.
        """
        if jacobian.shape[-2:] == (2, 3):
            measure = np.linalg.norm(
                np.cross(jacobian[..., 0, :], jacobian[..., 1, :]), axis=-1)
        else:
            measure = np.abs(np.linalg.det(jacobian))
        return measure * self.reference_element.integration_weights

    def _build_integration_jacobian(self, coords, elems):
        """Build Jacobians at the reference integration points.

        Parameters
        ----------
        coords : numpy.ndarray, shape (N_nodes, dim_phys)
            Physical coordinates of all mesh nodes, including unused nodes.
        elems : numpy.ndarray, shape (N_elems, N_points)
            Integer connectivity using indices into ``coords``.

        Returns
        -------
        jacobian : numpy.ndarray, shape (N_elems, N_quad, dim_ref, dim_phys)
            One Jacobian per element and quadrature point. Center Jacobians
            for gradient output are built by ``FiniteElementGradient``.
        """
        return np.einsum('qri,eip->eqrp',
                         self.reference_element.integration_dN, coords[elems],
                         optimize=True)
