import numpy as np


class LinearTriangleElement:
    """Class representing a linear triangular finite element.

    ``N1 = 1 - xi - eta``
    ``N2 = xi``
    ``N3 = eta``

    Attributes:
    ----------
    name: str
        Name of the element type.
    mass_coef: float
        Coefficient for mass matrix calculation.
    elem_mass: (3, 3) ndarray
        Element mass matrix.
    dN: (2, 3) ndarray
        Derivative of shape functions with respect to xi and eta.
    quad_weights: (1,) ndarray
        Quadrature weights for the element.
    n_points: int
        Number of points (nodes) in the element.
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
