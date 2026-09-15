import numpy as np
import pytest

import finitewave as fw


class DiffusionOnlyModel:
    D_model = 1.0

    def initialize(self, simulation):
        self._u = simulation.cardiac_tissue.coords[:, 0].copy()
        self._rhs = np.zeros_like(self._u)

    def run(self):
        self._rhs.fill(0.0)

    def sync_backend(self):
        pass


@pytest.mark.parametrize(
    ("element_type", "coords", "elems"),
    [
        (
            fw.ElementType.TRIANGLE,
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.array([[0, 1, 2]]),
        ),
        (
            fw.ElementType.QUAD,
            np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]),
            np.array([[0, 1, 2, 3]]),
        ),
        (
            fw.ElementType.TETRA,
            np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            np.array([[0, 1, 2, 3]]),
        ),
        (
            fw.ElementType.HEXAHEDRON,
            np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]),
            np.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
        ),
    ],
)
def test_compute_weights_smoke_for_all_element_types(
    element_type, coords, elems
):
    tissue = fw.CardiacTissueElements(coords, elems, element_type)

    stiffness, mass = fw.FiniteElementDiscretization().compute_weights(tissue)

    expected_shape = (coords.shape[0], coords.shape[0])
    assert stiffness.shape == expected_shape
    assert mass.shape == expected_shape
    assert np.all(np.isfinite(stiffness.data))
    assert np.all(np.isfinite(mass.data))
    np.testing.assert_allclose(stiffness.toarray(), stiffness.T.toarray())
    np.testing.assert_allclose(stiffness.sum(axis=1).A1, 0.0, atol=1e-12)
    assert np.all(mass.diagonal() > 0.0)


@pytest.mark.parametrize(
    ("element_type", "element_class"),
    [
        (fw.ElementType.TRIANGLE, fw.LinearTriangleElement),
        (fw.ElementType.QUAD, fw.LinearQuadrilateralElement),
        (fw.ElementType.TETRA, fw.LinearTetrahedralElement),
        (fw.ElementType.HEXAHEDRON, fw.LinearHexahedralElement),
    ],
)
def test_element_type_is_the_source_of_reference_element_names(
    element_type, element_class
):
    element = fw.ElementType.select_reference_element(element_type)

    assert isinstance(element, element_class)
    assert element.name == element_type
    assert fw.ElementType.is_valid(element.name)


def test_compute_weights_filters_diffusion_with_inactive_elements():
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    elems = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ])
    tissue = fw.CardiacTissueElements(
        coords, elems, fw.ElementType.TRIANGLE
    )
    tissue.conductivity = np.array([2.0, 7.0])
    tissue.mesh_elems[1] = 0

    discretization = fw.FiniteElementDiscretization()
    stiffness, mass = discretization.compute_weights(tissue)

    expected_diffusion = (2.0 * np.eye(2))[None, :, :]
    expected_stiffness, expected_mass = discretization.compute_system_matrices(
        coords, elems[:1], expected_diffusion
    )

    np.testing.assert_allclose(
        stiffness.toarray(), expected_stiffness.toarray()
    )
    np.testing.assert_allclose(mass.toarray(), expected_mass.toarray())
    np.testing.assert_allclose(stiffness.getrow(3).toarray(), 0.0)
    np.testing.assert_allclose(mass.getrow(3).toarray(), 0.0)


def test_fiber_direction_controls_anisotropic_stiffness():
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    elems = np.array([[0, 1, 2]])
    tissue = fw.CardiacTissueElements(
        coords, elems, fw.ElementType.TRIANGLE
    )
    tissue.fibers = np.array([[1.0, 0.0]])
    tissue.D_al = 4.0
    tissue.D_ac = 1.0
    tissue.conductivity = 0.5

    stiffness, _ = fw.FiniteElementDiscretization().compute_weights(tissue)

    expected = np.array([
        [1.25, -1.0, -0.25],
        [-1.0, 1.0, 0.0],
        [-0.25, 0.0, 0.25],
    ])
    np.testing.assert_allclose(stiffness.toarray(), expected)


def test_fem_runs_one_implicit_diffusion_step():
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    tissue = fw.CardiacTissueElements(
        coords, np.array([[0, 1, 2]]), fw.ElementType.TRIANGLE
    )
    simulation = fw.CardiacSimulation(
        dt=0.01, t_max=0.01, backend="numba"
    )
    simulation.cardiac_tissue = tissue
    simulation.cardiac_model = DiffusionOnlyModel()
    simulation.initialize()

    initial = simulation.cardiac_model._u.copy()
    mass = simulation.spatial_discretization.weights[1]
    initial_mass = np.ones(initial.size) @ mass @ initial
    initial_energy = initial @ mass @ initial

    simulation.run(initialize=False, prog_bar=False)

    result = simulation.cardiac_model._u
    result_mass = np.ones(result.size) @ mass @ result
    result_energy = result @ mass @ result
    assert simulation.iteration == 1
    assert np.all(np.isfinite(result))
    assert not np.allclose(result, initial)
    np.testing.assert_allclose(result_mass, initial_mass)
    assert result_energy < initial_energy
