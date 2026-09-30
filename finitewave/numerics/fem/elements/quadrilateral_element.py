import numpy as np

from .element_type import ElementType


class LinearQuadrilateralElement:
    """Bilinear quadrilateral on ``[-1, 1]^2``.

    Nodes are ordered (-1, -1), (1, -1), (1, 1), (-1, 1).

    Attributes
    ----------
    name : str
        Element type identifier.
    order : int
        Polynomial order (1).
    n_points : int
        Number of element nodes (4), not evaluation points.
    center : ndarray, shape (1, 2)
        Reference centroid.
    gauss_points : ndarray, shape (4, 2)
        Tensor Gauss integration points; each has reference weight 1.
    gauss_weights : ndarray, shape (4,)
        Weights corresponding to ``gauss_points``, summing to the reference
        area (4).

    Notes
    -----
    Evaluate shape functions and reference derivatives at arbitrary points
    using ``shape_function`` and ``shape_function_derivative``. The leading
    evaluation-point axis is retained even for a single point. Use
    ``gauss_points`` with ``gauss_weights`` for integration. Physical
    derivatives and integration measures require a geometry-dependent Jacobian.
    """

    name = ElementType.QUAD
    order = 1
    center = np.array([[0.0, 0.0]])
    gauss_points = 1 / np.sqrt(3) * np.array([[-1, -1],
                                              [1, -1],
                                              [1, 1],
                                              [-1, 1]])
    gauss_weights = np.full(4, 1.0)
    n_points = 4

    def shape_function(self, points):
        """Evaluate shape functions at supplied reference points.

        Parameters
        ----------
        points : array_like, shape (N_eval, 2) or (2,)
            Reference coordinates of evaluation points.

        Returns
        -------
        N : ndarray, shape (N_eval, 4)
            Shape-function values with axes (evaluation point, node).
        """
        points = np.atleast_2d(points)
        xi = points[:, 0]
        eta = points[:, 1]
        N = np.column_stack(((1 - xi) * (1 - eta) / 4,
                             (1 + xi) * (1 - eta) / 4,
                             (1 + xi) * (1 + eta) / 4,
                             (1 - xi) * (1 + eta) / 4))
        return N

    def shape_function_derivative(self, points):
        """Evaluate reference shape-function derivatives at supplied reference points.

        Parameters
        ----------
        points : array_like, shape (N_eval, 2) or (2,)
            Reference coordinates of evaluation points.

        Returns
        -------
        dN : ndarray, shape (N_eval, 2, 4)
            Derivatives with axes (evaluation point, reference direction, node).
        """
        points = np.atleast_2d(points)

        xi = points[:, 0]
        eta = points[:, 1]
        dN_dxi = np.column_stack((-(1 - eta),
                                   (1 - eta),
                                   (1 + eta),
                                  -(1 + eta))) / 4
    
        dN_deta = np.column_stack((-(1 - xi),
                                   -(1 + xi),
                                    (1 + xi),
                                    (1 - xi))) / 4
        dN = np.stack((dN_dxi, dN_deta), axis=1)
        return dN
