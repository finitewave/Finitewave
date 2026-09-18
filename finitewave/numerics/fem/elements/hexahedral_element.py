import numpy as np


class LinearHexahedralElement:
    """Trilinear hexahedral element with eight nodes in 3D space.

    The nodes are ordered around the bottom face first, followed by the
    corresponding nodes on the top face::

        7-------6
       /|      /|
      4-------5 |
      | 3-----|-2
      |/      |/
      0-------1

    The reference element occupies ``[-1, 1]^3``.  ``dN`` contains the
    derivatives of the shape functions at its centre for gradient output.
    Matrix assembly uses the eight-point tensor Gauss rule.

    Attributes
    ----------
    name : str
        Name of the element type.
    order : int
        Polynomial order of the element.
    mass_coef : float
        Coefficient for the consistent mass matrix.
    elem_mass : (8, 8) ndarray
        Reference consistent mass matrix normalized by reference volume.
        Retained for reference; assembly uses ``integration_N``.
    dN : (3, 8) ndarray
        Shape-function derivatives with respect to xi, eta, and zeta at the
        centre of the reference element.
    quad_weights : (1,) ndarray
        Legacy center-rule weight; assembly uses ``integration_weights``.
    n_points : int
        Number of nodes in the element.
    integration_points : numpy.ndarray, shape (N_quad, dim_ref)
        Eight-point tensor Gauss rule on [-1, 1]^3.
    integration_weights : numpy.ndarray, shape (N_quad,)
        Integration weights, summing to the reference area or volume.
    integration_N : numpy.ndarray, shape (N_quad, N_points)
        Shape-function values at the integration points.
    integration_dN : numpy.ndarray, shape (N_quad, dim_ref, N_points)
        Reference shape-function derivatives at the integration points.
    """

    def __init__(self):
        self.name = "Hexahedron"
        self.order = 1
        self.mass_coef = 216.0

        self.elem_mass = 1 / self.mass_coef * np.array([
            [8, 4, 2, 4, 4, 2, 1, 2],
            [4, 8, 4, 2, 2, 4, 2, 1],
            [2, 4, 8, 4, 1, 2, 4, 2],
            [4, 2, 4, 8, 2, 1, 2, 4],
            [4, 2, 1, 2, 8, 4, 2, 4],
            [2, 4, 2, 1, 4, 8, 4, 2],
            [1, 2, 4, 2, 2, 4, 8, 4],
            [2, 1, 2, 4, 4, 2, 4, 8],
        ])

        self.dN = 1 / 8 * np.array([
            [-1, 1, 1, -1, -1, 1, 1, -1],
            [-1, -1, 1, 1, -1, -1, 1, 1],
            [-1, -1, -1, -1, 1, 1, 1, 1],
        ])

        self.quad_weights = np.array([8.0])
        self.n_points = 8

        a = 1 / np.sqrt(3)
        signs = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ])
        self.integration_points = a * signs
        self.integration_weights = np.ones(8)
        factors = 1 + self.integration_points[:, None, :] * signs[None, :, :]
        self.integration_N = np.prod(factors, axis=2) / 8
        self.integration_dN = np.stack([
            signs[:, axis] * np.prod(np.delete(factors, axis, axis=2), axis=2) / 8
            for axis in range(3)
        ], axis=1)
