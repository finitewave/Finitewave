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
    derivatives of the shape functions at its centre, which is the
    one-point integration rule used by the finite-element discretization.

    Attributes
    ----------
    name : str
        Name of the element type.
    order : int
        Polynomial order of the element.
    mass_coef : float
        Coefficient for the consistent mass matrix.
    elem_mass : (8, 8) ndarray
        Consistent element mass matrix normalized by element volume.
    dN : (3, 8) ndarray
        Shape-function derivatives with respect to xi, eta, and zeta at the
        centre of the reference element.
    quad_weights : (1,) ndarray
        Weight of the one-point quadrature rule.
    n_points : int
        Number of nodes in the element.
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
