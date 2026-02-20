
import numpy as np


class LinearTetrahedralElement:
    """Class representing a linear tetrahedral finite element.

    ``N1 = 1 - xi - eta - zeta``
    ``N2 = xi``
    ``N3 = eta``
    ``N4 = zeta``

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
    dN_dzeta: (4,) ndarray
        Derivative of shape functions with respect to zeta.
    quad_weights: (1,) ndarray
        Quadrature weights for the element.
    n_points: int
        Number of points (nodes) in the element.
    """

    def __init__(self):
        self.mass_coef = 20.0

        self.elem_mass = 1 / self.mass_coef * np.array([[2, 1, 1, 1],
                                                        [1, 2, 1, 1],
                                                        [1, 1, 2, 1],
                                                        [1, 1, 1, 2]])
        self.dN_dxi = np.array([-1.0, 1.0, 0.0, 0.0])
        self.dN_deta = np.array([-1.0, 0.0, 1.0, 0.0])
        self.dN_dzeta = np.array([-1.0, 0.0, 0.0, 1.0])

        self.quad_weights = np.array([1.0/6.0])
        self.n_points = 4
