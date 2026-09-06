import numpy as np

import finitewave as fw


def _unit_cube():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    return coords, np.arange(8, dtype=np.int64)[None, :]


def test_linear_hexahedral_element_properties():
    element = fw.LinearHexahedralElement()

    assert element.name == fw.ElementType.HEXAHEDRON
    assert element.order == 1
    assert element.n_points == 8
    assert element.dN.shape == (3, 8)
    assert element.elem_mass.shape == (8, 8)
    np.testing.assert_allclose(element.dN.sum(axis=1), 0.0)
    np.testing.assert_allclose(element.elem_mass.sum(), 1.0)
    np.testing.assert_allclose(element.elem_mass, element.elem_mass.T)


def test_hexahedral_element_is_registered():
    element = fw.ElementType.select_reference_element(fw.ElementType.HEXAHEDRON)

    assert isinstance(element, fw.LinearHexahedralElement)
    assert fw.ElementType.HEXAHEDRON in fw.ElementType.volume


def test_unit_cube_metrics_and_system_matrices():
    coords, elems = _unit_cube()
    discretization = fw.FiniteElementDiscretization()
    discretization.reference_element = fw.LinearHexahedralElement()

    np.testing.assert_allclose(
        discretization.compute_elements_size(coords, elems),
        [1.0],
    )

    gradients = discretization.compute_gradients(coords, elems)
    np.testing.assert_allclose(gradients.sum(axis=2), 0.0)

    diffusion = np.eye(3)[None, :, :]
    stiffness, mass = discretization.compute_system_matrices(
        coords, elems, diffusion
    )
    np.testing.assert_allclose(stiffness.toarray().sum(axis=1), 0.0)
    np.testing.assert_allclose(mass.toarray().sum(), 1.0)


def test_build_hexahedral_slab_from_cubes():
    coords, elems = fw.build_hexahedral_slab(
        2, 1, 1,
        (0.0, 2.0),
        (0.0, 1.0),
        (0.0, 1.0),
    )

    assert coords.shape == (12, 3)
    assert elems.shape == (2, 8)

    discretization = fw.FiniteElementDiscretization()
    discretization.reference_element = fw.LinearHexahedralElement()
    np.testing.assert_allclose(
        discretization.compute_elements_size(coords, elems),
        [1.0, 1.0],
    )
