import numpy as np

from .element_type import ElementType


class LinearHexahedralElement:
    """Trilinear hexahedron on ``[-1, 1]^3``.

    Nodes run around the bottom face, then the corresponding top face::

        7-------6
       /|      /|
      4-------5 |
      | 3-----|-2
      |/      |/
      0-------1

    Attributes
    ----------
    name : str
        Element type identifier.
    order : int
        Polynomial order (1).
    n_points : int
        Number of element nodes (8), not evaluation points.
    center : ndarray, shape (1, 3)
        Reference centroid.
    gauss_points : ndarray, shape (8, 3)
        Tensor Gauss integration points; each has reference weight 1.
    gauss_weights : ndarray, shape (8,)
        Weights corresponding to ``gauss_points``, summing to the reference
        volume (8).

    Notes
    -----
    Evaluate shape functions and reference derivatives at arbitrary points
    using ``shape_function`` and ``shape_function_derivative``. The leading
    evaluation-point axis is retained even for a single point. Use
    ``gauss_points`` with ``gauss_weights`` for integration. Physical
    derivatives and integration measures require a geometry-dependent Jacobian.
    """

    name = ElementType.HEXAHEDRON
    order = 1
    center = np.array([[0.0, 0.0, 0.0]])
    gauss_points = 1 / np.sqrt(3) * np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ])
    gauss_weights = np.full(8, 1.0)
    n_points = 8

    def shape_function(self, points):
        """Evaluate shape functions at supplied reference points.

        Parameters
        ----------
        points : array_like, shape (N_eval, 3) or (3,)
            Reference coordinates of evaluation points.

        Returns
        -------
        N : ndarray, shape (N_eval, 8)
            Shape-function values with axes (evaluation point, node).
        """
        points = np.atleast_2d(points)
        xi = points[:, 0]
        eta = points[:, 1]
        zeta = points[:, 2]
        N = np.column_stack(((1 - xi) * (1 - eta) * (1 - zeta),
                             (1 + xi) * (1 - eta) * (1 - zeta),
                             (1 + xi) * (1 + eta) * (1 - zeta),
                             (1 - xi) * (1 + eta) * (1 - zeta),
                             (1 - xi) * (1 - eta) * (1 + zeta),
                             (1 + xi) * (1 - eta) * (1 + zeta),
                             (1 + xi) * (1 + eta) * (1 + zeta),
                             (1 - xi) * (1 + eta) * (1 + zeta))) / 8
        return N

    def shape_function_derivative(self, points):
        """Evaluate reference shape-function derivatives at supplied reference points.

        Parameters
        ----------
        points : array_like, shape (N_eval, 3) or (3,)
            Reference coordinates of evaluation points.

        Returns
        -------
        dN : ndarray, shape (N_eval, 3, 8)
            Derivatives with axes (evaluation point, reference direction, node).
        """
        points = np.atleast_2d(points)
        xi = points[:, 0]
        eta = points[:, 1]
        zeta = points[:, 2]
        dN_dxi = np.column_stack((-(1 - eta) * (1 - zeta),
                                   (1 - eta) * (1 - zeta),
                                   (1 + eta) * (1 - zeta),
                                  -(1 + eta) * (1 - zeta),
                                  -(1 - eta) * (1 + zeta),
                                   (1 - eta) * (1 + zeta),
                                   (1 + eta) * (1 + zeta),
                                  -(1 + eta) * (1 + zeta))) / 8

        dN_deta = np.column_stack((-(1 - xi) * (1 - zeta),
                                   -(1 + xi) * (1 - zeta),
                                    (1 + xi) * (1 - zeta),
                                    (1 - xi) * (1 - zeta),
                                   -(1 - xi) * (1 + zeta),
                                   -(1 + xi) * (1 + zeta),
                                    (1 + xi) * (1 + zeta),
                                    (1 - xi) * (1 + zeta))) / 8
        
        dN_dzeta = np.column_stack((-(1 - xi) * (1 - eta),
                                    -(1 + xi) * (1 - eta),
                                    -(1 + xi) * (1 + eta),
                                    -(1 - xi) * (1 + eta),
                                     (1 - xi) * (1 - eta),
                                     (1 + xi) * (1 - eta),
                                     (1 + xi) * (1 + eta),
                                     (1 - xi) * (1 + eta))) / 8
        
        dN = np.stack((dN_dxi, dN_deta, dN_dzeta), axis=1)
        return dN
