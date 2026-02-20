import numpy as np

from .volume_assembler import VolumeAssembler


class SurfaceAssembler(VolumeAssembler):
    """
    Class for assembling surface element diffusion models.

    Attributes
    ----------
    reference_element : ReferenceElement
        The reference element used for numerical integration.
    """

    def __init__(self):
        super().__init__()

    def compute_metrics(self, coords, elems):
        """
        Computes area and global gradients for surface elements in 3D
        using the Jacobian.

        Parameters:
        ----------
        coords: (N_nodes, 3)
            Coordinates of the mesh nodes.
        elems: (N_elems, N_points)
            Element connectivity (node indices for each surface element).

        Returns:
        -------
        areas: (N_elems,)
            Area of each surface element.
        grads: (N_elems, N_points, 3)
            Gradient of shape functions in global coordinates for each element.
        """
        jacobian = self.build_jacobian(coords, elems)
        areas = self._compute_areas(jacobian)
        grads = self._compute_gradients(jacobian)

        return areas, grads

    def build_jacobian(self, coords, elems):
        """
        Build Jacobian matrices for surface elements.

        Parameters:
        ----------
        coords: (N_nodes, 3)
            Coordinates of the mesh nodes.
        elems: (N_elems, N_points)
            Element connectivity (node indices for each surface element).

        Returns:
        -------
        jacobian: (N_elems, 2, 3)
            Jacobian matrices for each surface element.
        """
        n_elems = elems.shape[0]
        jacobian = np.zeros((n_elems, 2, 3))

        for i in range(self.reference_element.n_points):
            jacobian[:, 0, :] += (self.reference_element.dN_dxi[i] *
                                  coords[elems[:, i]])
            jacobian[:, 1, :] += (self.reference_element.dN_deta[i] *
                                  coords[elems[:, i]])

        return jacobian
    
    def compute_areas(self, coords, elems):
        """
        Computes areas of surface elements from their Jacobian matrices.

        Parameters:
        ----------
        coords: (N_nodes, 3)
            Coordinates of the mesh nodes.
        elems: (N_elems, N_points)
            Element connectivity (node indices for each surface element).

        Returns:
        -------
            areas: (N_elems,)
                Area of each surface element.
        """
        jacobian = self.build_jacobian(coords, elems)
        areas = self._compute_areas(jacobian)
        return areas

    def _compute_gradients(self, jacobian):
        """Compute global gradients for surface elements.

        Parameters:
        ----------
        jacobian: (N, 2, 3)
            Jacobian matrices for N surface elements.

        Returns:
        -------
            grads: (N, n_points, 3)
                Gradient of shape functions in global coordinates for each
                element.

        Note:
        -----
            The output shape is (N, n_points, 3) to match the expected format
            for subsequent computations.
        """
        jacobian_inv = self.invert_jacobian(jacobian)
        n_elems = jacobian_inv.shape[0]
        n_points = self.reference_element.n_points
        grads = np.zeros((n_elems, n_points, 3))

        for i in range(n_points):
            dN_ref = np.stack(
                [np.full(n_elems, self.reference_element.dN_dxi[i]),
                 np.full(n_elems, self.reference_element.dN_deta[i])], axis=1
                )
            grads[:, i, :] = np.einsum('nij,nj->ni', jacobian_inv, dN_ref)

        return grads

    def _compute_areas(self, jacobian):
        """Compute areas of surface elements from their Jacobian matrices.

        Parameters:
        ----------
        jacobian: (N, 2, 3)
            Jacobian matrices for N surface elements.

        Returns:
        -------
            areas: (N,)
                Area of each surface element.
        """
        v1 = jacobian[:, 0, :]
        v2 = jacobian[:, 1, :]
        cross_v = np.cross(v1, v2)
        areas = (self.reference_element.quad_weights *
                 np.linalg.norm(cross_v, axis=1))
        return areas

    def invert_jacobian(self, jacobian):
        """Compute the inverse of a 2x3 Jacobian using the pseudoinverse.

        Parameters:
        ----------
        jacobian: (N, 2, 3)
            Jacobian matrices for N elements.

        Returns:
        -------
            Jplus: (N, 3, 2)
                Right pseudoinverse of the Jacobian matrices.
        """
        JT = np.transpose(jacobian, (0, 2, 1))  # (N, 3, 2)
        G = np.matmul(jacobian, JT)             # (N, 2, 2)
        invG = np.linalg.inv(G)
        Jplus = np.matmul(JT, invG)  # (N, 3, 2) — right pseudoinverse
        return Jplus
