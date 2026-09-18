import numpy as np
import scipy.sparse as sp


class FiniteElementGradient:
    """Build center-gradient operators for surface and volume elements.

    Parameters
    ----------
    reference_element : object, optional
        Reference derivatives ``dN`` of shape (dim_ref, N_points) and node
        count ``n_points``. Set before calling ``build_gradient_operator``.

    Notes
    -----
    This class evaluates gradients at element centers. Quadrature-point
    geometry and gradients for stiffness assembly belong to
    ``FiniteElementDiffusion``. The module-level ``invert_jacobian`` helper
    supports both center and quadrature-point batches.
    """

    def __init__(self, reference_element=None):
        self.reference_element = reference_element

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
        jacobian = self._build_jacobian(coords, elems)
        grads = self._compute_gradient_operator(jacobian)

        if not as_sparse:
            return grads

        n_elems, dim_phys, n_points = grads.shape

        rows = np.repeat(np.arange(n_elems), n_points)
        cols = elems.ravel()
        shape = (n_elems, coords.shape[0])

        grads_ops = tuple(
            sp.coo_matrix((grads[:, axis, :].ravel(), (rows, cols)),
                            shape=shape,).tocsr()
            for axis in range(dim_phys)
        )
        return grads_ops

    def _build_jacobian(self, coords, elems):
        """
        Build center Jacobian matrices for elements.

        Parameters
        ----------
        coords : numpy.ndarray, shape (N_nodes, dim_phys)
            Coordinates of the mesh nodes.
        elems : numpy.ndarray, shape (N_elems, N_points)
            Element connectivity (node indices for each element).

        Returns
        -------
        jacobian : numpy.ndarray, shape (N_elems, dim_ref, dim_phys)
            Jacobian matrices for each element.
        """
        n_elems = elems.shape[0]
        dim_ref = len(self.reference_element.dN)
        dim_phys = coords.shape[1]
        jacobian = np.zeros((n_elems, dim_ref, dim_phys))

        for i in range(self.reference_element.n_points):
            for j in range(len(self.reference_element.dN)):
                jacobian[:, j, :] += (self.reference_element.dN[j, i] *
                                      coords[elems[:, i]])

        return jacobian

    def _compute_gradient_operator(self, jacobian):
        """Compute global shape-function gradients for elements.

        Parameters
        ----------
        jacobian : numpy.ndarray, shape (N_elems, dim_ref, dim_phys)
            Jacobian matrices for N elements.

        Returns
        -------
        grads : numpy.ndarray, shape (N_elems, dim_phys, N_points)
            Gradient of shape functions in global coordinates for each element.

        """
        n_elems, dim_ref, dim_phys = jacobian.shape
        jacobian_inv = invert_jacobian(jacobian)
        n_points = self.reference_element.n_points
        grads = np.zeros((n_elems, dim_phys, n_points))

        for i in range(n_points):
            dN_ref = np.stack(
                [np.full(n_elems, self.reference_element.dN[j, i]) for j in range(dim_ref)],
                axis=1
            )
            grads[:, :, i] = (jacobian_inv @ dN_ref[..., None])[..., 0]

        return grads


def invert_jacobian(jacobian):
    """Invert Jacobian matrices.

    Parameters
    ----------
    jacobian : numpy.ndarray, shape (..., dim_ref, dim_phys)
        Jacobians with element and optional quadrature-point batch axes.

    Returns
    -------
    jacobian_inv : numpy.ndarray, shape (..., dim_phys, dim_ref)
        Inverse for square Jacobians or right pseudoinverse for surface
        elements embedded in a higher-dimensional space.
    """
    if jacobian.shape[-2] == jacobian.shape[-1]:
        return np.linalg.inv(jacobian)

    # pseudo-inverse for non-square Jacobian (e.g., for surface elements)
    JT = np.swapaxes(jacobian, -1, -2)
    G = np.matmul(jacobian, JT)
    invG = np.linalg.inv(G)
    Jplus = np.matmul(JT, invG)
    return Jplus
