import numpy as np
from finitewave.core.numerics.fem.elements.reference_element import ReferenceElement
from finitewave.core.numerics.fem.elements.element_type import ElementShape, ElementOrder


class LinearTetrahedralElement(ReferenceElement):
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
    dN: (3, 4) ndarray
        Derivative of shape functions with respect to xi, eta, and zeta.
    quad_weights: (1,) ndarray
        Quadrature weights for the element.
    n_points: int
        Number of points (nodes) in the element.
    """

    def __init__(self):
        super().__init__()
        self.shape = ElementShape.TETRA
        self.order = ElementOrder.LINEAR
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

    