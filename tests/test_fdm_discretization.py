import numpy as np
import pytest

import finitewave as fw


@pytest.mark.parametrize(
    ("discretization", "scale"),
    [
        (fw.IsotropicDiscretization(), 2.0), # here we use 'ghost points' to treat the boundary
        (fw.AsymmetricDiscretization(), 1.0),
    ],
)
def test_boundary_conditions_assemble_2d(discretization, scale):
    mesh = np.pad(np.ones((2, 2), dtype=np.int8), 1)
    tissue = fw.CardiacTissue(mesh=mesh, dr=1.0)

    stiffness, mass = discretization.compute_weights(tissue)

    expected = scale * np.array(
        [
            [2, -1, -1, 0],
            [-1, 2, 0, -1],
            [-1, 0, 2, -1],
            [0, -1, -1, 2],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(stiffness.toarray(), expected)
    np.testing.assert_allclose(mass.toarray(), np.eye(4))


@pytest.mark.parametrize(
    ("discretization", "expected_diagonal"),
    [
        (fw.IsotropicDiscretization(), 6.0),
        (fw.AsymmetricDiscretization(), 3.0),
    ],
)
def test_boundary_conditions_assemble_3d(discretization, expected_diagonal):
    mesh = np.ones((2, 2, 2), dtype=np.int8)

    stiffness = discretization.compute_diffusion_operator(mesh, dr=1.0)

    np.testing.assert_allclose(stiffness.diagonal(), expected_diagonal)
    np.testing.assert_allclose(stiffness.sum(axis=1).A1, 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "discretization",
    [fw.IsotropicDiscretization(), fw.AsymmetricDiscretization()],
)
def test_fibrotic_cells_are_decoupled(discretization):
    mesh = np.ones((3, 3), dtype=np.int8)
    mesh[1, 1] = 2
    indexes = np.flatnonzero(mesh[mesh > 0] == 1)

    stiffness = discretization.compute_diffusion_operator(
        mesh, dr=1.0, indexes=indexes
    )

    np.testing.assert_allclose(stiffness.sum(axis=1).A1, 0.0, atol=1e-12)
    np.testing.assert_allclose(stiffness.getrow(4).toarray(), 0.0)
    np.testing.assert_allclose(stiffness.getcol(4).toarray(), 0.0)


@pytest.mark.parametrize(
    ("discretization", "expected"),
    [
        (
            fw.IsotropicDiscretization(),
            [[0.5, -0.5, 0.0], [-0.25, 1.0, -0.75], [0.0, -1.5, 1.5]],
        ),
        (
            fw.AsymmetricDiscretization(),
            [[0.25, -0.25, 0.0], [-0.25, 1.0, -0.75], [0.0, -0.75, 0.75]],
        ),
    ],
)
@pytest.mark.parametrize("compressed", [False, True])
def test_spatial_connectivity_is_attached_to_positive_edges(
    discretization, expected, compressed
):
    mesh = np.ones((1, 3), dtype=np.int8)
    connectivity = np.ones(mesh.shape + (mesh.ndim,))
    connectivity[0, 0, 1] = 0.25
    connectivity[0, 1, 1] = 0.75
    if compressed:
        connectivity = connectivity[mesh > 0]

    stiffness = discretization.compute_diffusion_operator(
        mesh, dr=1.0, connectivity=connectivity
    )

    np.testing.assert_allclose(stiffness.toarray(), expected)


@pytest.mark.parametrize(
    ("method", "left", "right"),
    [("arithmetic", 2.0, 4.0), ("harmonic", 1.5, 3.75)],
)
def test_diffusion_averaging_method(method, left, right):
    mesh = np.ones((1, 3), dtype=np.int8)
    diffusion = np.zeros((3, 2, 2))
    diffusion[:, 0, 0] = 1.0
    diffusion[:, 1, 1] = [1.0, 3.0, 5.0]

    stiffness = fw.AsymmetricDiscretization(method).compute_diffusion_operator(
        mesh, dr=1.0, diffusion=diffusion
    )

    expected = np.array(
        [[left, -left, 0.0], [-left, left + right, -right], [0.0, -right, right]]
    )
    np.testing.assert_allclose(stiffness.toarray(), expected)


def test_harmonic_average_of_zero_coefficients_is_finite():
    mesh = np.ones((1, 2), dtype=np.int8)
    diffusion = np.zeros((2, 2, 2))

    stiffness = fw.AsymmetricDiscretization(
        "harmonic"
    ).compute_diffusion_operator(mesh, dr=1.0, diffusion=diffusion)

    assert np.all(np.isfinite(stiffness.data))
    np.testing.assert_allclose(stiffness.toarray(), 0.0)


def test_constant_anisotropic_tensor_has_expected_interior_stencil():
    mesh = np.ones((3, 3), dtype=np.int8)
    diffusion = np.array([[2.0, 0.6], [0.6, 1.0]])

    stiffness = fw.AsymmetricDiscretization().compute_diffusion_operator(
        mesh, dr=1.0, diffusion=diffusion
    )

    expected_center_stencil = np.array(
        [[-0.3, -2.0, 0.3], [-1.0, 6.0, -1.0], [0.3, -2.0, -0.3]]
    )
    np.testing.assert_allclose(
        stiffness.getrow(4).toarray().reshape(3, 3), expected_center_stencil
    )
    np.testing.assert_allclose(stiffness.sum(axis=1).A1, 0.0, atol=1e-12)


def test_tissue_fibers_feed_the_asymmetric_operator():
    mesh = np.pad(np.ones((3, 3), dtype=np.int8), 1)
    tissue = fw.CardiacTissue(mesh=mesh, dr=1.0)
    tissue.D_al = 2.0
    tissue.D_ac = 1.0
    tissue.fibers = np.zeros(mesh.shape + (mesh.ndim,))
    tissue.fibers[..., 0] = np.sqrt(0.5)
    tissue.fibers[..., 1] = np.sqrt(0.5)

    stiffness, _ = fw.AsymmetricDiscretization().compute_weights(tissue)

    expected_center_stencil = np.array(
        [[-0.25, -1.5, 0.25], [-1.5, 6.0, -1.5], [0.25, -1.5, -0.25]]
    )
    np.testing.assert_allclose(
        stiffness.getrow(4).toarray().reshape(3, 3), expected_center_stencil
    )


@pytest.mark.parametrize(
    "discretization",
    [fw.IsotropicDiscretization(), fw.AsymmetricDiscretization()],
)
def test_full_grid_and_tissue_indexed_fields_are_equivalent(discretization):
    mesh = np.pad(np.ones((2, 2), dtype=np.int8), 1)
    ndim = mesh.ndim
    diffusion = np.zeros(mesh.shape + (ndim, ndim))
    diffusion[..., 0, 0] = np.arange(mesh.size).reshape(mesh.shape) + 1
    diffusion[..., 1, 1] = 2.0
    connectivity = np.ones(mesh.shape + (ndim,))
    connectivity[..., 0] = 0.75

    full_grid = discretization.compute_diffusion_operator(
        mesh, dr=0.5, diffusion=diffusion, connectivity=connectivity
    )
    compressed = discretization.compute_diffusion_operator(
        mesh,
        dr=0.5,
        diffusion=diffusion[mesh > 0],
        connectivity=connectivity[mesh > 0],
    )

    np.testing.assert_allclose(full_grid.toarray(), compressed.toarray())


def test_isotropic_scalar_flux_has_one_weight_per_node():
    mesh = np.ones((3, 3), dtype=np.int8)
    indexes = np.arange(mesh.size)
    ijk = np.array(np.unravel_index(indexes, mesh.shape))

    _, weights = fw.IsotropicDiscretization()._flux_weights(
        mesh=mesh,
        diffusion=0.5,
        connectivity=1.0,
        dr=0.25,
        ijk=ijk,
        axis=0,
        tissue_index_map=indexes.reshape(mesh.shape),
    )

    assert all(weight.shape == (mesh.size,) for weight in weights)
    np.testing.assert_allclose(weights[0], 2.0)
    np.testing.assert_allclose(weights[1], -2.0)
    np.testing.assert_allclose(weights[2], 2.0)
    np.testing.assert_allclose(weights[3], -2.0)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("diffusion", np.ones((3, 3, 3))),
        ("connectivity", np.ones((3, 3, 3))),
    ],
)
def test_invalid_field_shapes_are_rejected(argument, value):
    kwargs = {argument: value}

    with pytest.raises(ValueError):
        fw.AsymmetricDiscretization().compute_diffusion_operator(
            np.ones((2, 2), dtype=np.int8), dr=1.0, **kwargs
        )


def test_nonpositive_grid_spacing_is_rejected():
    with pytest.raises(ValueError, match="dr must be positive"):
        fw.IsotropicDiscretization().compute_diffusion_operator(
            np.ones((2, 2), dtype=np.int8), dr=0.0
        )


@pytest.mark.parametrize(
    "discretization",
    [fw.IsotropicDiscretization(), fw.AsymmetricDiscretization()],
)
def test_empty_active_region_builds_an_empty_operator(discretization):
    stiffness = discretization.compute_diffusion_operator(
        np.zeros((2, 2), dtype=np.int8), dr=1.0
    )

    assert stiffness.shape == (0, 0)
    assert stiffness.nnz == 0
