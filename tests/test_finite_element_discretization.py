import numpy as np
import pytest

import finitewave as fw


def _element_sizes(diffusion, coords, elems):
    jacobian = diffusion._build_integration_jacobian(coords, elems)
    return diffusion._compute_integration_weights(jacobian).sum(axis=1)


@pytest.mark.parametrize("embedded", [False, True])
def test_mass_matrix_on_two_triangles_with_unused_node(embedded):
    coords = np.array([
        [0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 2.],
    ])
    if embedded:
        coords = np.column_stack((coords, np.zeros(len(coords))))
    elems = np.array([[0, 1, 2], [1, 3, 2]])
    discretization = fw.FiniteElementDiscretization()
    discretization.reference_element = fw.LinearTriangleElement()

    # Each triangle has area 1/2; shared nodes receive both contributions.
    expected = np.array([
        [2, 1, 1, 0, 0],
        [1, 4, 2, 1, 0],
        [1, 2, 4, 1, 0],
        [0, 1, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ]) / 24.
    mass = discretization.build_mass_matrix(coords, elems)
    assert mass.format == "csr"
    np.testing.assert_allclose(mass.toarray(), expected)

    stiffness, combined_mass = discretization.build_system_matrices(
        coords, elems, diffusion=2.)
    np.testing.assert_allclose(combined_mass.toarray(), expected)
    np.testing.assert_allclose(
        discretization.build_diffusion_operator(
            coords, elems, diffusion=2.).toarray(),
        stiffness.toarray(),
    )


ELEMENT_CASES = [
    (fw.LinearTriangleElement, [[0, 0], [1, 0], [0, 1]], 0.5),
    (fw.LinearTetrahedralElement,
     [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], 1/6),
    (fw.LinearQuadrilateralElement, [[0, 0], [1, 0], [1, 1], [0, 1]], 1.),
    (fw.LinearHexahedralElement,
     [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
      [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], 1.),
]


@pytest.mark.parametrize("element_class, nodes, volume", ELEMENT_CASES)
def test_quadrature_on_affine_elements(element_class, nodes, volume):
    reference = element_class()
    coords = np.array(nodes, dtype=float)
    dim = coords.shape[1]
    # Shear and stretch to exercise non-diagonal Jacobians.
    transform = np.eye(dim) * 2
    transform[0, 1] = 0.4
    coords = coords @ transform + 0.7
    volume *= np.linalg.det(transform)
    elems = np.arange(len(coords))[None, :]
    discretization = fw.FiniteElementDiscretization()
    discretization.reference_element = reference
    diffusion = np.eye(dim) + 0.2 * np.ones((dim, dim))
    stiffness, mass = discretization.build_system_matrices(
        coords, elems, diffusion[None, :, :])
    K, M = stiffness.toarray(), mass.toarray()

    np.testing.assert_allclose(reference.integration_N.sum(axis=1), 1)
    np.testing.assert_allclose(reference.integration_dN.sum(axis=2), 0, atol=1e-15)
    np.testing.assert_allclose(M, volume * reference.elem_mass, atol=1e-14)
    np.testing.assert_allclose(_element_sizes(discretization.diffusion, coords, elems), [volume])
    np.testing.assert_allclose(K, K.T, atol=1e-14)
    np.testing.assert_allclose(K.sum(axis=1), 0, atol=1e-14)
    eigenvalues = np.linalg.eigvalsh(K)
    assert abs(eigenvalues[0]) < 1e-12
    assert np.all(eigenvalues[1:] > 1e-8)

    # Exact energy of a physical linear field under anisotropic diffusion.
    gradient = np.arange(1, dim + 1, dtype=float)
    values = coords @ gradient + 3
    np.testing.assert_allclose(values @ K @ values,
                               volume * gradient @ diffusion @ gradient)
    np.testing.assert_allclose(discretization.build_mass_matrix(coords, elems).toarray(), M)
    np.testing.assert_allclose(discretization.build_diffusion_operator(
        coords, elems, diffusion[None, :, :]).toarray(), K)

    dense = discretization.build_gradient_operator(coords, elems, as_sparse=False)
    sparse = discretization.build_gradient_operator(coords, elems)
    np.testing.assert_allclose(dense[0] @ values, gradient)
    np.testing.assert_allclose([op @ values for op in sparse], gradient[:, None])


def test_square_stiffness_and_alternating_mode():
    discretization = fw.FiniteElementDiscretization()
    discretization.reference_element = fw.LinearQuadrilateralElement()
    coords = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    elems = np.array([[0, 1, 2, 3]])
    stiffness = discretization.build_diffusion_operator(coords, elems).toarray()
    expected = np.array([[4, -1, -2, -1], [-1, 4, -1, -2],
                         [-2, -1, 4, -1], [-1, -2, -1, 4]]) / 6
    np.testing.assert_allclose(stiffness, expected)
    values = np.array([1., -1., 1., -1.])
    np.testing.assert_allclose(values @ stiffness @ values, 8/3)


def test_embedded_quad_rotation_preserves_matrices():
    discretization = fw.FiniteElementDiscretization()
    discretization.reference_element = fw.LinearQuadrilateralElement()
    coords = np.array([[0., 0.], [2., 0.], [1.5, 1.], [0., 1.]])
    elems = np.array([[0, 1, 2, 3]])
    K, M = discretization.build_system_matrices(coords, elems)
    # Rotate the planar trapezoid into three dimensions.
    basis = np.array([[1., 0., 0.], [0., 0.6, 0.8]])
    embedded = coords @ basis + [2., 3., 4.]
    K3, M3 = discretization.build_system_matrices(embedded, elems)
    np.testing.assert_allclose(K3.toarray(), K.toarray(), atol=1e-14)
    np.testing.assert_allclose(M3.toarray(), M.toarray(), atol=1e-14)
    np.testing.assert_allclose(_element_sizes(discretization.diffusion, embedded, elems), [1.75])
    np.testing.assert_allclose(M.sum(), 1.75)
    # Non-affine Jacobians vary between integration points.
    J = discretization.diffusion._build_integration_jacobian(embedded, elems)
    assert not np.allclose(J[:, 0], J[:, 2])


def test_tapered_hexahedron_integrates_varying_jacobian():
    discretization = fw.FiniteElementDiscretization()
    discretization.reference_element = fw.LinearHexahedralElement()
    coords = np.array([
        [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
        [0., 0., 1.], [2., 0., 1.], [2., 2., 1.], [0., 2., 1.],
    ])
    elems = np.arange(8)[None, :]
    # Cross-section area is (1 + z)**2; its integral from 0 to 1 is 7/3.
    volume = 7/3
    K, M = discretization.build_system_matrices(coords, elems)
    np.testing.assert_allclose(_element_sizes(discretization.diffusion, coords, elems), [volume])
    np.testing.assert_allclose(M.sum(), volume)
    gradient = np.array([1., 2., 3.])
    values = coords @ gradient
    np.testing.assert_allclose(values @ K @ values, volume * (gradient @ gradient))


@pytest.mark.parametrize("element_class, nodes, volume", ELEMENT_CASES)
def test_direct_operators_match_discretization(element_class, nodes, volume):
    coords = np.array(nodes, dtype=float)
    elems = np.arange(len(coords))[None, :]
    reference = element_class()
    discretization = fw.FiniteElementDiscretization()
    discretization.reference_element = reference
    diffusion = fw.FiniteElementDiffusion(reference)
    gradient = fw.FiniteElementGradient(reference)
    expected_K, expected_M = discretization.build_system_matrices(coords, elems)
    K = diffusion.build_diffusion_operator(coords, elems)
    M = discretization.build_mass_matrix(coords, elems)
    np.testing.assert_allclose(K.toarray(), expected_K.toarray())
    np.testing.assert_allclose(M.toarray(), expected_M.toarray())
    np.testing.assert_allclose(
        gradient.build_gradient_operator(coords, elems, as_sparse=False),
        discretization.build_gradient_operator(coords, elems, as_sparse=False),
    )
    np.testing.assert_allclose(_element_sizes(diffusion, coords, elems), [volume])


def test_reference_element_updates_both_operators_and_tissue_weights():
    from types import SimpleNamespace

    discretization = fw.FiniteElementDiscretization()
    # Reuse the same facade across element types and physical dimensions.
    for element_class, nodes, volume in ELEMENT_CASES:
        reference = element_class()
        coords = np.array(nodes, dtype=float)
        elems = np.arange(len(coords))[None, :]
        tissue = SimpleNamespace(
            coords=coords, myo_elems=elems, reference_element=reference,
            diffusion_tensor=np.eye(coords.shape[1])[None, :, :],
        )
        K, M = discretization.compute_weights(tissue, D_model=2.)
        assert discretization.reference_element is reference
        assert discretization.diffusion.reference_element is reference
        assert discretization.gradient.reference_element is reference
        np.testing.assert_allclose(M.sum(), volume)
        np.testing.assert_allclose(K.toarray(),
                                   discretization.build_diffusion_operator(coords, elems, 2.).toarray())
        gradients = discretization.build_gradient_operator(coords, elems, as_sparse=False)
        np.testing.assert_allclose(gradients[0] @ coords, np.eye(coords.shape[1]), atol=1e-14)
