import numpy as np

from finitewave.core.numerics.fem.elements.element_type import ElementShape, ElementOrder

class ReferenceElement:
    """
    Abstract base class for finite elements.
    Use as a contract for all finite element types 
    to implement the required attributes.
    """
    shape: ElementShape
    order: ElementOrder
    n_points: int
    elem_mass: np.ndarray
    dN: np.ndarray
    quad_weights: np.ndarray
