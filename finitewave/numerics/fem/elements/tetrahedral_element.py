import numpy as np

from .element_type import ElementType


class LinearTetrahedralElement:
    """Linear tetrahedron with nodes (0, 0, 0), (1, 0, 0),
    (0, 1, 0), and (0, 0, 1).

    Shape functions are ``1 - xi - eta - zeta``, ``xi``, ``eta``, ``zeta``.

    Attributes
    ----------
    name : str
        Element type identifier.
    order : int
        Polynomial order (1).
    n_points : int
        Number of element nodes (4), not evaluation points.
    center : ndarray, shape (1, 3)
        Reference centroid.
    gauss_points : ndarray, shape (4, 3)
        Degree-two simplex integration points; each has reference weight 1/24.
    gauss_weights : ndarray, shape (4,)
        Weights corresponding to ``gauss_points``, summing to the reference
        volume (1/6).

    Notes
    -----
    Evaluate shape functions and reference derivatives at arbitrary points
    using ``shape_function`` and ``shape_function_derivative``. The leading
    evaluation-point axis is retained even for a single point. Use
    ``gauss_points`` with ``gauss_weights`` for integration. Physical
    derivatives and integration measures require a geometry-dependent Jacobian.
    """

    name = ElementType.TETRA
    order = 1
    center = np.array([[1/4, 1/4, 1/4]])
    _a = (5 + 3 * np.sqrt(5)) / 20
    _b = (5 - np.sqrt(5)) / 20
    gauss_points = np.array([[_b, _b, _b],
                             [_a, _b, _b],
                             [_b, _a, _b],
                             [_b, _b, _a]])
    gauss_weights = np.full(4, 1/24)
    n_points = 4

    def shape_function(self, points):
        """Evaluate shape functions at supplied reference points.

        Parameters
        ----------
        points : array_like, shape (N_eval, 3) or (3,)
            Reference coordinates of evaluation points.

        Returns
        -------
        N : ndarray, shape (N_eval, 4)
            Shape-function values with axes (evaluation point, node).
        """
        points = np.atleast_2d(points)
        xi, eta, zeta = points.T
        return np.column_stack((1 - xi - eta - zeta, xi, eta, zeta))

    def shape_function_derivative(self, points):
        """Evaluate reference shape-function derivatives at supplied reference points.

        Parameters
        ----------
        points : array_like, shape (N_eval, 3) or (3,)
            Reference coordinates of evaluation points.

        Returns
        -------
        dN : ndarray, shape (N_eval, 3, 4)
            Derivatives with axes (evaluation point, reference direction, node).
        """
        points = np.atleast_2d(points)
        dN = np.array([[-1.0, 1.0, 0.0, 0.0],
                       [-1.0, 0.0, 1.0, 0.0],
                       [-1.0, 0.0, 0.0, 1.0]])
        return np.repeat(dN[None, :, :], points.shape[0], axis=0)
