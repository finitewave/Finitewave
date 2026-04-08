
import numpy as np
import scipy.sparse as sp
from numba import njit, prange


class ElementAssembler:
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
        Computes area/volume and global gradients for 3D elements
        using the Jacobian from the reference element.

        Parameters
        ----------
        coords : (N_nodes, dim_phys)
            Coordinates of the mesh nodes.
        elems : (N_elems, N_points)
            Element connectivity (node indices for each element).

        Returns
        --------
        elements_size : (N_elems,)
            Area or volume of each element.
        grads : (N_elems, N_points, dim_phys)
            Gradient of shape functions in global coordinates for each element.
        """
        jacobian = self.build_jacobian(coords, elems)
        elements_size = self._compute_elements_size(jacobian)
        grads = self._compute_gradients(jacobian)

        return elements_size, grads

    def build_jacobian(self, coords, elems):
        """
        Build Jacobian matrices for elements.

        Parameters
        ----------
        coords : (N_nodes, dim_phys)
            Coordinates of the mesh nodes.
        elems : (N_elems, N_points)
            Element connectivity (node indices for each element).

        Returns
        -------
        jacobian : (N_elems, dim_ref, dim_phys)
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
            
        # for e in range(n_elems):
        #     try:
        #         np.linalg.inv(jacobian[e])
        #     except np.linalg.LinAlgError:
        #         print(f"Singular Jacobian for element {e}. Check the mesh quality.")
        #         print(f"Element connectivity:\n{elems[e]}")
        #         print(f"Element nodes:\n{coords[elems[e]]}")
        #         raise ValueError()
        return jacobian
    
    def compute_gradients(self, coords, elems):
        """
        Compute global gradients for elements.

        Parameters:
        ----------
        coords : (N_nodes, dim_phys)
            Coordinates of the mesh nodes.
        elems : (N_elems, N_points)
            Element connectivity (node indices for each element).

        Returns:
        -------
            grads: (N_elems, dim_phys, N_points)
                Gradient of shape functions in global coordinates for each
                element.
        """
        jacobian = self.build_jacobian(coords, elems)
        grads = self._compute_gradients(jacobian)
        # Transpose to (N_elems, dim_phys, N_points)
        # grads = np.transpose(grads, (0, 2, 1)).copy() 
        return grads
    
    def compute_elements_size(self, coords, elems):
        """
        Compute area/volume of elements.

        Parameters:
        ----------
        jacobian: (N_elems, dim_ref, dim_phys)
            Jacobian matrices for N elements.

        Returns:
        -------
            elements_size: (N_elems,)
                Area or volume of each element.
        """
        jacobian = self.build_jacobian(coords, elems)
        return self._compute_elements_size(jacobian)

    def _compute_gradients(self, jacobian):
        """Compute global gradients for triangle elements.

        Parameters:
        ----------
        jacobian: (N_elems, dim_ref, dim_phys)
            Jacobian matrices for N triangle elements.

        Returns:
        -------
            grads: (N_elems, dim_phys, N_points)
                Gradient of shape functions in global coordinates for each
                element.

        Note:
            The output shape is (N_elems, dim_phys, N_points) to match the expected format
            for subsequent computations.
        """
        n_elems, dim_ref, dim_phys = jacobian.shape
        jacobian_inv = self.invert_jacobian(jacobian)
        n_points = self.reference_element.n_points
        grads = np.zeros((n_elems, dim_phys, n_points))

        for i in range(n_points):
            dN_ref = np.stack(
                [np.full(n_elems, self.reference_element.dN[j, i]) for j in range(dim_ref)],
                axis=1
            )
            grads[:, :, i] = (jacobian_inv @ dN_ref[..., None])[..., 0]

        return grads

    def _compute_elements_size(self, jacobian):
        """Compute areas/volumes of elements from their Jacobian matrices.

        Parameters:
        ----------
        jacobian: (N_elems, dim_ref, dim_phys)
            Jacobian matrices for N elements.

        Returns:
        -------
            elements_size: (N_elems,)
                Area or volume of each element.
        """
        if jacobian.shape[1] == 2:
            v1 = jacobian[:, 0, :]
            v2 = jacobian[:, 1, :]
            if v1.shape[1] != 3:
                v1 = np.hstack([v1, np.zeros((v1.shape[0], 1))])
            if v2.shape[1] != 3:
                v2 = np.hstack([v2, np.zeros((v2.shape[0], 1))])

            cross_prod = np.linalg.norm(np.cross(v1, v2), axis=1)
            return self.reference_element.quad_weights * cross_prod
        
        jacobian_det = np.abs(np.linalg.det(jacobian))
        return self.reference_element.quad_weights * jacobian_det

    def invert_jacobian(self, jacobian):
        """Invert Jacobian matrices.

        Parameters:
        ----------
        jacobian: (N_elems, dim_ref, dim_phys)
            Jacobian matrices for N elements.

        Returns:
        -------
            jacobian_inv: (N_elems, dim_phys, dim_ref)
                Inverted Jacobian matrices for N elements.
        """
        if jacobian.shape[1] == jacobian.shape[2]:
            return np.linalg.inv(jacobian)
        
        # pseudo-inverse for non-square Jacobian (e.g., for surface elements)
        JT = np.transpose(jacobian, (0, 2, 1))  # (N, 3, 2)
        G = np.matmul(jacobian, JT)             # (N, 2, 2)
        invG = np.linalg.inv(G)
        Jplus = np.matmul(JT, invG)             # (N, 3, 2) — right pseudoinverse
        return Jplus
   
    def compute_system_matrices(self, coords, elems, diffusion, indexes, reindex=False):
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
        elem_mass = self.reference_element.elem_mass

        if reindex:
            elems = self.reindex_elems(coords, elems, indexes)
            coords = coords[indexes]
            shape = (len(indexes), len(indexes))

        elements_size, grads = self.compute_metrics(coords, elems)
        # TODO: replace with einsum for better performance and readability
        # rows, cols, stiff, mass = stiffness_and_mass_matrix(elems, elements_size,
        #                                                     grads, diffusion,
        #                                                     elem_mass)
        # stiffness_matrix = sp.coo_matrix((stiff, (rows, cols)), shape=shape)
        # mass_matrix = sp.coo_matrix((mass, (rows, cols)), shape=shape)

        stiff_matrix, mass_matrix = self._stiffness_and_mass_matrix(elems, elements_size,
                                                                    grads, diffusion,
                                                                    elem_mass, shape)

        # n_components, labels = sp.csgraph.connected_components(stiffness_matrix, directed=False)
        # print(f"Connected components in the stiffness matrix: {n_components}")
        return stiff_matrix, mass_matrix
    
    def _stiffness_and_mass_matrix(self, elems, elems_size, grads, diffusion, elem_mass, shape):
        """
        Computes the stiffness matrix.

        K[e] = V[e] * (grad[e, i] @ D[e] @ grad[e, j])

        Parameters
        ----------
        elems : np.ndarray [N_elems, N_points]
            The connectivity of the mesh elements.
        elems_size : np.ndarray [N_elems,]
            The volumes/areas of the elements.
        grads : np.ndarray [N_elems, N_points, dim_phys]
            The gradients of the shape functions.
        diffusion : np.ndarray [N_elems, dim_phys, dim_phys]
            The diffusion tensors for each element.
        elem_mass : np.ndarray [N_points, N_points]
            The mass matrix for each element.

        Returns
        -------
        sp.csr_matrix
            The stiffness matrix in sparse format.
        """
        n_elems, n_points = elems.shape
        rows = np.repeat(elems[:, :, np.newaxis], n_points, axis=2).flatten()
        cols = np.repeat(elems[:, np.newaxis, :], n_points, axis=1).flatten()

        stiff_data = np.einsum('e,eki,elk,elj->eij', elems_size, grads, diffusion, grads, optimize='optimal')
        stiff_data = stiff_data.flatten()

        mass_data = np.einsum('e,ij->eij', elems_size, elem_mass, optimize='optimal')
        mass_data = mass_data.flatten()

        stiff_matrix = sp.coo_matrix((stiff_data, (rows, cols)), shape=shape)
        mass_matrix = sp.coo_matrix((mass_data, (rows, cols)), shape=shape)
        return stiff_matrix.tocsr(), mass_matrix.tocsr()
    
    def reindex_elems(self, coords, elems, indexes):
        """
        Reindexes the element connectivity array. Resulted indexes are
        continuous and start from 0.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the mesh nodes.
        elems : np.ndarray
            The connectivity of the mesh elements.
        indexes : np.ndarray
            The indexes of the nodes in the original mesh.

        Returns
        -------
        np.ndarray
            The reindexed element connectivity array.
        """
        new_indexes = - np.ones(coords.shape[0], dtype=int)
        new_indexes[indexes] = np.arange(len(indexes))
        new_elems = new_indexes[elems]
        if np.any(new_elems < 0):
            raise ValueError("Some elements are not connected to the specified nodes.")
        return new_elems


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
                ind = n_points ** 2 * e + n_points * i + j
                rows[ind] = elems[e, i]
                cols[ind] = elems[e, j]
                val = volumes[e] * (grads[e, i, :] @
                                    diffusion[e] @
                                    grads[e, j, :])
                stiff_data[ind] = val
                mass_data[ind] = volumes[e] * elem_mass[i, j]

    return rows, cols, stiff_data, mass_data
