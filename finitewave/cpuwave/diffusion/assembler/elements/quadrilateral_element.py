import numpy as np


class LinearQuadrilateralElement:
    """Linear quadrilateral element with 4 nodes in 3D space.

    ``N1 = 0.25 * (1 - xi) * (1 - eta)``
    ``N2 = 0.25 * (1 + xi) * (1 - eta)``
    ``N3 = 0.25 * (1 + xi) * (1 + eta)``
    ``N4 = 0.25 * (1 - xi) * (1 + eta)``

    Attributes:
    ----------
    mass_coef: float
        Coefficient for mass matrix calculation.
    elem_mass: (4, 4) ndarray
        Element mass matrix.
    dN_dxi: (4,) ndarray
        Derivative of shape functions with respect to xi.
    dN_deta: (4,) ndarray
        Derivative of shape functions with respect to eta.
    quad_weights: (1,) ndarray
        Quadrature weights for the element.
    n_points: int
        Number of points (nodes) in the element.
    """
    def __init__(self):
        super().__init__()
        self.mass_coef = 36.  # consistent with 4-node quads

        self.elem_mass = 1 / self.mass_coef * np.array([[4, 2, 1, 2],
                                                        [2, 4, 2, 1],
                                                        [1, 2, 4, 2],
                                                        [2, 1, 2, 4]])

        self.dN_dxi = np.array([-0.25,  0.25,  0.25, -0.25])
        self.dN_deta = np.array([-0.25, -0.25,  0.25,  0.25])
        self.quad_weights = np.array([4.0])
        self.n_points = 4
