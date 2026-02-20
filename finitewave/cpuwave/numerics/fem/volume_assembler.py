
import numpy as np
import scipy.sparse as sp
from numba import njit, prange


class VolumeAssembler:
    """
    Class for assembling volume element diffusion models.

    Attributes
    ----------
    reference_element : ReferenceElement
        The reference element used for numerical integration.
    """
    def __init__(self):
        pass

    def compute_metrics(self, coords, elems):
        """
        Computes volume and global gradients for 3D elements
        using the Jacobian from the reference element.

        Parameters:
        ----------
        coords: (N_nodes, 3)
            Coordinates of the mesh nodes.
        elems: (N_elems, N_points)
            Element connectivity (node indices for each element).

        Returns:
        -------
        volumes: (N_elems,)
            Volume of each element.
        grads: (N_elems, 3, 3)
            Gradient of shape functions in global coordinates for each element.
        """
        jacobian = self.build_jacobian(coords, elems)
        volumes = self._compute_volumes(jacobian)
        grads = self._compute_gradients(jacobian)

        return volumes, grads

    def build_jacobian(self, coords, elems):
        """
        Build Jacobian matrices for quadrilateral elements.

        Parameters:
        ----------
        coords: (N_nodes, 3)
            Coordinates of the mesh nodes.
        elems: (N_elems, N_points)
            Element connectivity (node indices for each quadrilateral).

        Returns:
        -------
        jacobian: (N_elems, 3, 3)
            Jacobian matrices for each quadrilateral element.
        """
        n_elems = elems.shape[0]
        jacobian = np.zeros((n_elems, 3, 3))

        for i in range(self.reference_element.n_points):
            jacobian[:, 0, :] += (self.reference_element.dN_dxi[i] *
                                  coords[elems[:, i]])
            jacobian[:, 1, :] += (self.reference_element.dN_deta[i] *
                                  coords[elems[:, i]])
            jacobian[:, 2, :] += (self.reference_element.dN_dzeta[i] *
                                  coords[elems[:, i]])

        return jacobian
    
    def compute_gradients(self, coords, elems):
        """
        Compute global gradients for quadrilateral elements.

        Parameters:
        ----------
        jacobian: (N_elems, 3, 3)
            Jacobian matrices for N quadrilateral elements.

        Returns:
        -------
            grads: (N_elems, 3, N_points)
                Gradient of shape functions in global coordinates for each
                element.
        """
        jacobian = self.build_jacobian(coords, elems)
        grads = self._compute_gradients(jacobian)
        grads = np.transpose(grads, (0, 2, 1)).copy()  # Transpose to (N_elems, 3, N_points)
        return grads
    
    def compute_volumes(self, coords, elems):
        """
        Compute volumes of quadrilateral elements from their Jacobian matrices.

        Parameters:
        ----------
        jacobian: (N_elems, 3, 3)
            Jacobian matrices for N quadrilateral elements.

        Returns:
        -------
            volumes: (N_elems,)
                Volume of each element.
        """
        jacobian = self.build_jacobian(coords, elems)
        return self._compute_volumes(jacobian)

    def _compute_gradients(self, jacobian):
        """Compute global gradients for triangle elements.

        Parameters:
        ----------
        jacobian: (N_elems, 3, 3)
            Jacobian matrices for N triangle elements.

        Returns:
        -------
            grads: (N_elems, N_points, 3)
                Gradient of shape functions in global coordinates for each
                element.

        Note:
            The output shape is (N_elems, N_points, 3) to match the expected format
            for subsequent computations.
        """
        jacobian_inv = self.invert_jacobian(jacobian)
        n_elems = jacobian_inv.shape[0]
        n_points = self.reference_element.n_points
        grads = np.zeros((n_elems, n_points, 3))

        for i in range(n_points):
            dN_ref = np.stack(
                [np.full(n_elems, self.reference_element.dN_dxi[i]),
                 np.full(n_elems, self.reference_element.dN_deta[i]),
                 np.full(n_elems, self.reference_element.dN_dzeta[i])], axis=1
                )
            grads[:, i, :] = np.matmul(jacobian_inv, dN_ref[:, :, np.newaxis]
                                       ).squeeze()

        return grads

    def _compute_volumes(self, jacobian):
        """Compute volumes of elements from their Jacobian matrices.

        Parameters:
        ----------
        jacobian: (N_elems, 3, 3)
            Jacobian matrices for N elements.

        Returns:
        -------
            volumes: (N_elems,)
                Volume of each element.
        """
        jacobian_det = np.abs(np.linalg.det(jacobian))
        return self.reference_element.quad_weights * jacobian_det

    def invert_jacobian(self, jacobian):
        """Invert Jacobian matrices.

        Parameters:
        ----------
        jacobian: (N_elems, 3, 3)
            Jacobian matrices for N elements.

        Returns:
        -------
            jacobian_inv: (N_elems, 3, 3)
                Inverted Jacobian matrices for N elements.
        """
        jacobian_inv = np.linalg.inv(jacobian)
        return jacobian_inv
   
    def compute_system_matrices(self, coords, elems, diffusion, reindex=False):
        """
        Computes the stiffness and mass matrices.
        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the mesh nodes.
        elems : np.ndarray
            The connectivity of the mesh elements.
        diffusion : np.ndarray
            The diffusion tensors for each element.
        reindex : bool, optional
            Whether to reindex the elements based on the myo indexes, by default False.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The rows, columns, stiffness matrix, and mass matrix.
        """
        shape = (coords.shape[0], coords.shape[0])
        indexes = self.simulation.cardiac_tissue.myo_indexes
        elem_mass = self.reference_element.elem_mass

        if reindex:
            elems = self.reindex_elems(coords, elems, indexes)

        volumes, grads = self.compute_metrics(coords, elems)
        rows, cols, stiff, mass = stiffness_and_mass_matrix(elems, volumes,
                                                            grads, diffusion,
                                                            elem_mass)
        stiffness_matrix = sp.coo_matrix((stiff, (rows, cols)), shape=shape)
        mass_matrix = sp.coo_matrix((mass, (rows, cols)), shape=shape)
        return stiffness_matrix.tocsr(), mass_matrix.tocsr()


@njit(parallel=True)
def stiffness_and_mass_matrix(elems, volumes, grads, diffusion, elem_mass):
    """
    Computes the stiffness and mass matrices.

    Parameters
    ----------
    elems : np.ndarray
        The connectivity of the mesh elements.
    volumes : np.ndarray
        The volumes/areas of the elements.
    grads : np.ndarray
        The gradients of the shape functions.
    diffusion : np.ndarray
        The diffusion tensors for each element.
    elem_mass : np.ndarray
        The mass matrix for each element.

    Returns
    -------
    np.ndarray
        The row indices of the stiffness and mass matrices.
    np.ndarray
        The column indices of the stiffness and mass matrices.
    np.ndarray
        The stiffness matrix data.
    np.ndarray
        The mass matrix data.
    """
    n_elems, n_points = elems.shape
    rows = np.zeros(n_elems * n_points ** 2, dtype=elems.dtype)
    cols = np.zeros_like(rows)
    stiff_data = np.zeros_like(rows, dtype=diffusion.dtype)
    mass_data = np.zeros_like(rows, dtype=diffusion.dtype)

    for e in prange(n_elems):
        for i in range(n_points):
            for j in range(n_points):
                ind = n_points ** 2 * (e - 1) + n_points * (i - 1) + j
                rows[ind] = elems[e, i]
                cols[ind] = elems[e, j]
                val = volumes[e] * (grads[e, i, :] @
                                    diffusion[e] @
                                    grads[e, j, :])
                stiff_data[ind] = val
                mass_data[ind] = volumes[e] * elem_mass[i, j]

    return rows, cols, stiff_data, mass_data
