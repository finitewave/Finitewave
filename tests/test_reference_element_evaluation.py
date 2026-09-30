"""Check reference interpolation and derivatives independently of assembly."""

import numpy as np
import pytest

from finitewave.numerics.fem.elements.element_type import ElementType


@pytest.mark.parametrize('name,nodes', [
    (ElementType.TRIANGLE, [[0, 0], [1, 0], [0, 1]]),
    (ElementType.QUAD, [[-1, -1], [1, -1], [1, 1], [-1, 1]]),
    (ElementType.TETRA, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    (ElementType.HEXAHEDRON, [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ]),
])
def test_reference_element_evaluation(name, nodes):
    element = ElementType.select_reference_element(name)
    nodes = np.asarray(nodes, dtype=float)
    dim = nodes.shape[1]
    np.testing.assert_allclose(element.shape_function(nodes), np.eye(len(nodes)))

    points = np.vstack((element.center, element.gauss_points, 0.7 * element.center))
    N = element.shape_function(points)
    dN = element.shape_function_derivative(points)
    assert N.shape == (len(points), element.n_points)
    assert dN.shape == (len(points), dim, element.n_points)
    np.testing.assert_allclose(N.sum(axis=1), 1)
    np.testing.assert_allclose(dN.sum(axis=2), 0, atol=1e-15)
    np.testing.assert_allclose(N @ nodes, points, atol=1e-15)
    np.testing.assert_allclose(
        dN @ nodes, np.broadcast_to(np.eye(dim), (len(points), dim, dim)),
        atol=1e-15,
    )

    step = 1e-6
    for axis in range(dim):
        offset = np.eye(dim)[axis] * step
        numerical = (element.shape_function(points + offset)
                     - element.shape_function(points - offset)) / (2 * step)
        np.testing.assert_allclose(dN[:, axis, :], numerical, atol=1e-9)

    assert element.shape_function(element.center[0]).shape == (1, len(nodes))
    assert element.shape_function_derivative(element.center[0]).shape == (1, dim, len(nodes))
