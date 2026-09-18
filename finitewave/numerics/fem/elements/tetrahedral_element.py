
import numpy as np


class LinearTetrahedralElement:
    """Class representing a linear tetrahedral finite element.

    ``N1 = 1 - xi - eta - zeta``
    ``N2 = xi``
    ``N3 = eta``
    ``N4 = zeta``

    Attributes
    ----------
    name: str
        Name of the element type.
    mass_coef: float
        Normalization coefficient for the reference ``elem_mass`` matrix.
    elem_mass: (4, 4) ndarray
        Reference consistent mass matrix normalized by reference area or
        volume. Retained for reference; assembly uses ``integration_N``.
    dN: (3, 4) ndarray
        Reference shape-function derivatives at the element center.
    quad_weights: (1,) ndarray
        Legacy center-rule weight; assembly uses ``integration_weights``.
    n_points: int
        Number of points (nodes) in the element.
    integration_points : numpy.ndarray, shape (N_quad, dim_ref)
        Four-point degree-two tetrahedron rule.
    integration_weights : numpy.ndarray, shape (N_quad,)
        Integration weights, summing to the reference area or volume.
    integration_N : numpy.ndarray, shape (N_quad, N_points)
        Shape-function values at the integration points.
    integration_dN : numpy.ndarray, shape (N_quad, dim_ref, N_points)
        Reference shape-function derivatives at the integration points.
    """

    def __init__(self):
        self.name = "Tetrahedral"
        self.order = 1
        self.mass_coef = 20.0

        self.elem_mass = 1 / self.mass_coef * np.array([[2, 1, 1, 1],
                                                        [1, 2, 1, 1],
                                                        [1, 1, 2, 1],
                                                        [1, 1, 1, 2]])
        self.dN = np.array([[-1.0, 1.0, 0.0, 0.0],
                            [-1.0, 0.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0, 1.0]])

        self.quad_weights = np.array([1.0/6.0])
        self.n_points = 4

        a = (5 + 3 * np.sqrt(5)) / 20
        b = (5 - np.sqrt(5)) / 20
        self.integration_N = np.full((4, 4), b)
        np.fill_diagonal(self.integration_N, a)
        self.integration_points = self.integration_N[:, 1:].copy()
        self.integration_weights = np.full(4, 1/24)
        self.integration_dN = np.repeat(self.dN[None, :, :], 4, axis=0)
