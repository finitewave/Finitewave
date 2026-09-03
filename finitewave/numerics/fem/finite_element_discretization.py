
import numpy as np
import scipy.sparse as sp

from finitewave.core.numerics.spatial_discretization import SpatialDiscretization


class FiniteElementDiscretization(SpatialDiscretization):
    """
    Class for assembling volume element diffusion models.

    Attributes
    ----------
    reference_element : ReferenceElement
        The reference element used for numerical integration.
    """
    def __init__(self):
        self.reference_element = None

    def compute_weights(self, tissue):
        """
        Computes the weights for the diffusion operator.

        Parameters
        ----------
        tissue : CardiacTissueBase
            The tissue object containing the mesh and diffusion tensor.

        Returns
        -------
        sparse.csr_matrix
            The stiffness matrix with shape (non_empty_nodes, non_empty_nodes).
        sparse.csr_matrix
            The mass matrix with shape (non_empty_nodes, non_empty_nodes).
        """
        diffusion = tissue.diffusion_tensor
        coords = tissue.coords
        elems = tissue.myo_elems
        self.reference_element = tissue.reference_element
        return self.compute_system_matrices(coords, elems, diffusion)

    def compute_system_matrices(self, coords, elems, diffusion=1.):
        """
        Computes the stiffness and mass matrices.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the mesh nodes.
        elems : np.ndarray
            The connectivity of the mesh elements.
        diffusion : np.ndarray, optional
            The diffusion tensors for each element, by default 1.

        Returns
        -------
        sparse.csr_matrix
            The stiffness matrix with shape (non_empty_nodes, non_empty_nodes).
        sparse.csr_matrix
            The mass matrix with shape (non_empty_nodes, non_empty_nodes).
        """
        n_elem_points = elems.shape[1]
        shape = (coords.shape[0], coords.shape[0])
        rows = np.repeat(elems, n_elem_points, axis=1).ravel()
        cols = np.tile(elems, (1, n_elem_points)).ravel()

        jacobian = self.build_jacobian(coords, elems)
        elems_size = self._compute_elements_size(jacobian)
        grads = self._compute_gradients(jacobian)
        stiffness = self._compute_diffusion_operator(rows, cols, elems_size, grads, diffusion, shape)
        mass = self._compute_mass_matrix(rows, cols, elems_size, self.reference_element.elem_mass, shape)
        return stiffness, mass

    def compute_diffusion_operator(self, coords, elems, diffusion=1.):
        """
        Computes the stiffness and mass matrices.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the mesh nodes.
        elems : np.ndarray
            The connectivity of the mesh elements.
        indexes : np.ndarray, optional
            The indexes of the non-empty nodes in the mesh, by default None.
        diffusion : np.ndarray, optional
            The diffusion tensors for each element, by default 1.
        conductivity : np.ndarray, optional
            The conductivity values for each element, by default None.

        Returns
        -------
        sparse.csr_matrix
            The stiffness matrix with shape (non_empty_nodes, non_empty_nodes).
        """
        n_points = coords.shape[0]
        shape = (n_points, n_points)
        rows = np.repeat(elems, n_points, axis=1).ravel()
        cols = np.tile(elems, (1, n_points)).ravel()

        jacobian = self.build_jacobian(coords, elems)
        elems_size = self._compute_elements_size(jacobian)
        grads = self._compute_gradients(jacobian)
        stiffness = self._compute_diffusion_operator(rows, cols, elems_size, grads, diffusion, shape)
        return stiffness

    def compute_mass_matrix(self, coords, elems):
        """
        Computes the mass matrix.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the mesh nodes.
        elems : np.ndarray
            The connectivity of the mesh elements.

        Returns
        -------
        sparse.csr_matrix
            The mass matrix with shape (non_empty_nodes, non_empty_nodes).
        """
        n_points = coords.shape[0]
        shape = (n_points, n_points)
        rows = np.repeat(elems, n_points, axis=1).ravel()
        cols = np.tile(elems, (1, n_points)).ravel()

        jacobian = self.build_jacobian(coords, elems)
        elems_size = self._compute_elements_size(jacobian)
        mass = self._compute_mass_matrix(rows, cols, elems_size, self.reference_element.elem_mass, shape)
        return mass

    def _compute_mass_matrix(self, rows, cols, elems_size, elem_mass, shape):
        mass_data = np.einsum('e,ij->eij', elems_size, elem_mass, optimize='optimal')
        mass_data = mass_data.flatten()
        mass_matrix = sp.coo_matrix((mass_data, (rows, cols)), shape=shape)
        return mass_matrix.tocsr()

    def _compute_diffusion_operator(self, rows, cols, elems_size, grads, diffusion, shape):
        stiff_data = np.einsum('e,eki,elk,elj->eij', elems_size, grads, diffusion, grads, optimize='optimal')
        stiff_data = stiff_data.flatten()
        stiff_matrix = sp.coo_matrix((stiff_data, (rows, cols)), shape=shape)
        return stiff_matrix.tocsr()

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
