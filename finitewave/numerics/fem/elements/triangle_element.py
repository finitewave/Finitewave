import numpy as np
from .element_type import ElementType


class LinearTriangleElement:
    """Linear triangle with nodes (0, 0), (1, 0), and (0, 1).

    Shape functions are ``1 - xi - eta``, ``xi``, and ``eta``.

    Attributes
    ----------
    name : str
        Element type identifier.
    order : int
        Polynomial order (1).
    n_points : int
        Number of element nodes (3), not evaluation points.
    center : ndarray, shape (1, 2)
        Reference centroid.
    gauss_points : ndarray, shape (3, 2)
        Degree-two simplex integration points; each has reference weight 1/6.
    gauss_weights : ndarray, shape (3,)
        Weights corresponding to ``gauss_points``, summing to the reference
        area (1/2).

    Notes
    -----
    Evaluate shape functions and reference derivatives at arbitrary points
    using ``shape_function`` and ``shape_function_derivative``. The leading
    evaluation-point axis is retained even for a single point. Use
    ``gauss_points`` with ``gauss_weights`` for integration. Physical
    derivatives and integration measures require a geometry-dependent Jacobian.
    """

    name = ElementType.TRIANGLE
    order = 1
    center = np.array([[1/3, 1/3]])
    gauss_points = np.array([[1/6, 1/6],
                             [2/3, 1/6],
                             [1/6, 2/3]])
    gauss_weights = np.full(3, 1/6)
    n_points = 3

    def shape_function(self, points):
        """Evaluate shape functions at supplied reference points.

        Parameters
        ----------
        points : array_like, shape (N_eval, 2) or (2,)
            Reference coordinates of evaluation points.

        Returns
        -------
        N : ndarray, shape (N_eval, 3)
            Shape-function values with axes (evaluation point, node).
        """
        points = np.atleast_2d(points)
        xi = points[:, 0]
        eta = points[:, 1]
        N = np.column_stack((1 - xi - eta, xi, eta))
        return N

    def shape_function_derivative(self, points):
        """Evaluate reference shape-function derivatives at supplied reference points.

        Parameters
        ----------
        points : array_like, shape (N_eval, 2) or (2,)
            Reference coordinates of evaluation points.

        Returns
        -------
        dN : ndarray, shape (N_eval, 2, 3)
            Derivatives with axes (evaluation point, reference direction, node).
        """
        points = np.atleast_2d(points)
        dN = np.array([[-1.0, 1.0, 0.0],
                       [-1.0, 0.0, 1.0]])
        return np.repeat(dN[None, :, :], points.shape[0], axis=0)
