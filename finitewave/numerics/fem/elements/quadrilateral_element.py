import numpy as np


class LinearQuadrilateralElement:
    """Bilinear quadrilateral element with four nodes in 2D or 3D space.

    ``N1 = 0.25 * (1 - xi) * (1 - eta)``
    ``N2 = 0.25 * (1 + xi) * (1 - eta)``
    ``N3 = 0.25 * (1 + xi) * (1 + eta)``
    ``N4 = 0.25 * (1 - xi) * (1 + eta)``

    Attributes
    ----------
    name: str
        Name of the element type.
    mass_coef: float
        Normalization coefficient for the reference ``elem_mass`` matrix.
    elem_mass: (4, 4) ndarray
        Reference consistent mass matrix normalized by reference area or
        volume. Retained for reference; assembly uses ``integration_N``.
    dN: (2, 4) ndarray
        Reference shape-function derivatives at the element center.
    quad_weights: (1,) ndarray
        Legacy center-rule weight; assembly uses ``integration_weights``.
    n_points: int
        Number of points (nodes) in the element.
    integration_points : numpy.ndarray, shape (N_quad, dim_ref)
        Four-point tensor Gauss rule on [-1, 1]^2.
    integration_weights : numpy.ndarray, shape (N_quad,)
        Integration weights, summing to the reference area or volume.
    integration_N : numpy.ndarray, shape (N_quad, N_points)
        Shape-function values at the integration points.
    integration_dN : numpy.ndarray, shape (N_quad, dim_ref, N_points)
        Reference shape-function derivatives at the integration points.
    """
    def __init__(self):
        super().__init__()
        self.name = "Quadrilateral"
        self.order = 1
        self.mass_coef = 36.  # consistent with 4-node quads

        self.elem_mass = 1 / self.mass_coef * np.array([[4, 2, 1, 2],
                                                        [2, 4, 2, 1],
                                                        [1, 2, 4, 2],
                                                        [2, 1, 2, 4]])

        self.dN = np.array([[-0.25,  0.25,  0.25, -0.25],
                            [-0.25, -0.25,  0.25,  0.25]])
        self.quad_weights = np.array([4.0])
        self.n_points = 4

        signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        self.integration_points = (1 / np.sqrt(3)) * signs
        self.integration_weights = np.ones(4)

        factors = 1 + self.integration_points[:, None, :] * signs[None, :, :]
        self.integration_N = np.prod(factors, axis=2) / 4
        self.integration_dN = np.stack([
            signs[:, 0] * factors[:, :, 1] / 4,
            signs[:, 1] * factors[:, :, 0] / 4,
        ], axis=1)
