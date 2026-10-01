from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from finitewave import ECGTracker, LeadFieldECGTracker
from finitewave.simulation.tissue.cardiac_tissue_grid import CardiacTissueGrid


@pytest.fixture(params=["numba", "jax", "mlx"])
def backend(request):
    pytest.importorskip(request.param)
    if request.param == "numba":
        from finitewave.numerics.backends.numba_backend import NumbaBackend
        return NumbaBackend()
    if request.param == "jax":
        from finitewave.numerics.backends.jax_backend import JAXBackend
        return JAXBackend()
    from finitewave.numerics.backends.mlx_backend import MlxBackend
    return MlxBackend()


@pytest.mark.parametrize("tracker_class", [ECGTracker, LeadFieldECGTracker])
@pytest.mark.parametrize("geometry", ["grid2d", "grid3d", "elements"])
@pytest.mark.parametrize("ratio", [2.0, 3.5])
def test_ecg_matches_dense_bilinear_form(backend, geometry, tracker_class, ratio, tmp_path):
    # Row 1 is empty; grid indexing is not contiguous.
    K = sp.csr_matrix([[2., 0., -2.], [0., 0., 0.], [-2., 0., 2.]])
    u = np.array([1., 50., 4.])
    if geometry.startswith("grid"):
        shape = (3, 4) if geometry == "grid2d" else (2, 3, 4)
        mesh = np.zeros(shape, dtype=int)
        mesh.flat[[1, 5, 9]] = [1, 2, 1]
        tissue = CardiacTissueGrid(mesh=mesh, dr=0.2)
        positions = np.column_stack(np.unravel_index([1, 5, 9], shape))
        positions = np.pad(positions, ((0, 0), (0, 3 - positions.shape[1])))
        electrodes = np.array([[1., 2., 6.], [3., 2., 5.]])
        physical_positions = positions * tissue.dr
        physical_electrodes = electrodes * tissue.dr
    else:
        positions = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 1.]])
        tissue = SimpleNamespace(meta={"type": "Elements"}, tissue_coords=positions,
                                 myo_indexes=np.array([0, 2]))
        electrodes = np.array([[1., 2., 6.], [3., 2., 5.]])
        physical_positions, physical_electrodes = positions, electrodes

    simulation = SimpleNamespace(
        backend=backend, cardiac_tissue=tissue,
        cardiac_model=SimpleNamespace(_u=backend.wrap_array(u)),
        spatial_discretization=SimpleNamespace(weights=(K, sp.eye(3))),
        dt=0.01, t_max=1.)
    tracker = tracker_class(electrodes, volume_conductivity=2., min_distance=0.1,
                            mono_to_intra_ratio=ratio)
    tracker.initialize(simulation)
    distances = np.linalg.norm(
        physical_positions[None, :, :] - physical_electrodes[:, None, :], axis=-1)
    lead = 1. / distances
    volume = tissue.dr**3 if geometry.startswith("grid") else 1.0
    expected = -ratio * volume * (lead @ K.toarray() @ u) / (8 * np.pi)
    np.testing.assert_allclose(tracker.calc_ecg(), expected, rtol=3e-6, atol=1e-6)

    # The direct stiffness expression is independent of the time step.
    simulation.dt = 0.2
    tracker._track()
    simulation.cardiac_model._u = backend.wrap_array(np.ones(3))
    tracker._track()
    np.testing.assert_allclose(tracker.output[0], expected, rtol=3e-6, atol=1e-6)
    np.testing.assert_allclose(tracker.output[1], 0., atol=1e-6)
    tracker.write(tmp_path)
    np.testing.assert_array_equal(np.load(tmp_path / tracker.file_name), tracker.output)


def test_ecg_zero_operator_and_coordinate_padding(backend):
    tissue = CardiacTissueGrid(shape=(2, 2), dr=0.2)
    simulation = SimpleNamespace(
        backend=backend, cardiac_tissue=tissue,
        cardiac_model=SimpleNamespace(_u=backend.wrap_array(np.ones(4))),
        spatial_discretization=SimpleNamespace(weights=(sp.csr_matrix((4, 4)), sp.eye(4))),
        dt=0.01, t_max=1.)
    tracker = ECGTracker([5., 5.])
    tracker.initialize(simulation)
    np.testing.assert_array_equal(tracker.calc_ecg(), [0.])
    assert tracker._min_distance == 0.1


@pytest.mark.parametrize("tracker_class", [ECGTracker, LeadFieldECGTracker])
def test_ecg_reinitialize_rebuilds_default_operator(backend, tracker_class):
    tissue = CardiacTissueGrid(shape=(2, 2), dr=0.2)
    K = sp.eye(4, format="csr")
    simulation = SimpleNamespace(
        backend=backend, cardiac_tissue=tissue,
        cardiac_model=SimpleNamespace(_u=backend.wrap_array(np.ones(4))),
        spatial_discretization=SimpleNamespace(weights=(K, K.copy())),
        dt=0.01, t_max=1.)
    tracker = tracker_class([5., 5.])
    tracker.initialize(simulation)
    first = np.asarray(tracker.calc_ecg()).copy()
    np.testing.assert_array_equal(K.toarray(), np.eye(4))
    simulation.spatial_discretization.weights = (3 * K, K)
    tissue.dr = 0.4
    tracker.initialize(simulation)
    np.testing.assert_allclose(tracker.calc_ecg(), 12 * first, rtol=3e-6)
    assert tracker._min_distance == 0.2
    assert tracker.min_distance is None
    assert tracker.diffusion_operator is None
    tracker.mono_to_intra_ratio = 3.0
    tracker.initialize(simulation)
    np.testing.assert_allclose(tracker.calc_ecg(), 18 * first, rtol=3e-6)


def test_supplied_lead_fields_use_matvec_and_preserve_input(backend):
    tissue = CardiacTissueGrid(mesh=np.array([[1, 2], [1, 1]]), dr=0.2)
    K = sp.csr_matrix([[2., 0., -2., 0.], [0., 7., 0., 0.],
                       [-2., 0., 2., 0.], [0., 0., 0., 0.]])
    u = np.array([1., 50., 4., 3.])
    fields = np.array([[1., 2., 3., 4.], [-1., 2., 1., 2.]])
    original = fields.copy()
    simulation = SimpleNamespace(
        backend=backend, cardiac_tissue=tissue,
        cardiac_model=SimpleNamespace(_u=backend.wrap_array(u)),
        spatial_discretization=SimpleNamespace(weights=(K, sp.eye(4))),
        dt=0.01, t_max=1.)
    tracker = LeadFieldECGTracker(lead_fields=fields, volume_conductivity=7.)
    tracker.initialize(simulation)
    q = -2.0 * (K @ u)
    np.testing.assert_allclose(tracker.calc_ecg(), fields @ q, rtol=3e-6)
    np.testing.assert_array_equal(fields, original)
    tracker._track()
    simulation.spatial_discretization.weights = (2 * K, sp.eye(4))
    tracker.initialize(simulation)
    assert tracker.ecg == []
    np.testing.assert_allclose(tracker.calc_ecg(), 2 * (fields @ q), rtol=3e-6)


@pytest.mark.parametrize("dimension", [2, 3], ids=["triangles", "tetrahedra"])
@pytest.mark.parametrize("mode", ["on_demand", "from_coords", "supplied_fields"])
def test_element_ecg_matches_integrated_linear_lead_field(backend, dimension, mode):
    """FEM stiffness already includes element measure; do not multiply it twice.

    Independently integrate -grad(l_h).D_intra.grad(u_h) on each linear simplex.
    This references the nodal interpolant l_h of 1/r, as used by the tracker,
    rather than claiming exact integration of the continuous 1/r kernel.
    """
    from math import factorial
    from finitewave import CardiacTissueElements, ElementType, FiniteElementDiscretization

    # Unequal element sizes and a shared face exercise geometric weighting
    # and assembly. The final node is unused and has a deliberately large u.
    if dimension == 2:
        coords = np.array([[0., 0.], [2., 0.], [0., 1.],
                           [2., 3.], [7., 7.]])
        elems = np.array([[0, 1, 2], [1, 3, 2]])
        elem_type = ElementType.TRIANGLE
    else:
        coords = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.],
                           [0., 0., 3.], [2., 2., 3.], [7., 7., 7.]])
        elems = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        elem_type = ElementType.TETRA

    tissue = CardiacTissueElements(coords, elems, elem_type=elem_type)
    tissue.conductivity = np.array([1.3, 2.1])
    discretization = FiniteElementDiscretization()
    model_scale = 0.7
    K, M = discretization.compute_weights(tissue, D_model=model_scale)
    u = np.arange(len(coords), dtype=float)**2 + 0.5
    u[-1] = 1000.
    electrodes = np.array([[4., -2., 6.], [-3., 4., 5.]])
    positions = np.pad(coords, ((0, 0), (0, 3 - dimension)))
    conductivity = 2.3
    fields = 1. / (4 * np.pi * conductivity * np.linalg.norm(
        positions[None, :, :] - electrodes[:, None, :], axis=2))

    expected = np.zeros(len(electrodes))
    for elem, diffusion in zip(elems, tissue.conductivity):
        vertices = coords[elem]
        # Columns of inv([1, x, y, z]) are affine basis coefficients.
        basis_gradients = np.linalg.inv(
            np.column_stack((np.ones(len(elem)), vertices)))[1:, :]
        measure = abs(np.linalg.det(vertices[1:] - vertices[0])) / factorial(dimension)
        grad_u = basis_gradients @ u[elem]
        grad_leads = fields[:, elem] @ basis_gradients.T
        expected -= 2.0 * measure * model_scale * diffusion * (grad_leads @ grad_u)

    simulation = SimpleNamespace(
        backend=backend, cardiac_tissue=tissue,
        cardiac_model=SimpleNamespace(_u=backend.wrap_array(u)),
        spatial_discretization=SimpleNamespace(weights=(K, M)),
        dt=0.01, t_max=1.)
    if mode == "supplied_fields":
        tracker = LeadFieldECGTracker(lead_fields=fields)
    else:
        tracker_class = ECGTracker if mode == "on_demand" else LeadFieldECGTracker
        tracker = tracker_class(lead_coords=electrodes, volume_conductivity=conductivity)
    tracker.initialize(simulation)
    np.testing.assert_allclose(tracker.calc_ecg(), expected, rtol=1e-5, atol=1e-7)

    # A spatially constant voltage has no diffusion source, even with an
    # unused node and nonuniform element conductivity.
    simulation.cardiac_model._u = backend.wrap_array(np.ones(len(coords)))
    np.testing.assert_allclose(tracker.calc_ecg(), 0., atol=1e-7)
