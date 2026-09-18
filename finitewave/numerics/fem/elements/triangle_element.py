import numpy as np


class LinearTriangleElement:
    """Class representing a linear triangular finite element.

    ``N1 = 1 - xi - eta``
    ``N2 = xi``
    ``N3 = eta``

    Attributes
    ----------
    name: str
        Name of the element type.
    mass_coef: float
        Normalization coefficient for the reference ``elem_mass`` matrix.
    elem_mass: (3, 3) ndarray
        Reference consistent mass matrix normalized by reference area or
        volume. Retained for reference; assembly uses ``integration_N``.
    dN: (2, 3) ndarray
        Reference shape-function derivatives at the element center.
    quad_weights: (1,) ndarray
        Legacy center-rule weight; assembly uses ``integration_weights``.
    n_points: int
        Number of points (nodes) in the element.
    integration_points : numpy.ndarray, shape (N_quad, dim_ref)
        Three-point degree-two triangle rule.
    integration_weights : numpy.ndarray, shape (N_quad,)
        Integration weights, summing to the reference area or volume.
    integration_N : numpy.ndarray, shape (N_quad, N_points)
        Shape-function values at the integration points.
    integration_dN : numpy.ndarray, shape (N_quad, dim_ref, N_points)
        Reference shape-function derivatives at the integration points.
    """

    def __init__(self):
        self.name = "Triangle"
        self.order = 1
        self.mass_coef = 12.
        self.elem_mass = (1 / self.mass_coef) * np.array([[2, 1, 1],
                                                          [1, 2, 1],
                                                          [1, 1, 2]])

        self.dN = np.array([[-1.0, 1.0, 0.0],
                            [-1.0, 0.0, 1.0]])
        self.quad_weights = np.array([1/2])
        self.n_points = 3

        self.integration_points = np.array([[1/6, 1/6],
                                            [2/3, 1/6],
                                            [1/6, 2/3]])
        self.integration_weights = np.array([1/6, 1/6, 1/6])
        self.integration_N = np.column_stack((
            1 - self.integration_points.sum(axis=1), self.integration_points,
        ))
        self.integration_dN = np.repeat(self.dN[None, :, :], 3, axis=0)
