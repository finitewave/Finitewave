import numpy as np
import pytest

import finitewave as fw


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('operator', [fw.AsymmetricDiffusion, fw.IsotropicDiffusion])
def test_diffusion_scalar_tensor_and_boundary_stencil(ndim, operator):
    mesh = np.ones((3,) * ndim, dtype=np.int8)
    dr = 0.5
    assembler = fw.FiniteDifferenceDiscretization()
    assembler.diffusion = operator()
    K, M = assembler.build_system_matrices(mesh, dr, diffusion=2.)
    tensor = np.repeat((2 * np.eye(ndim))[None], mesh.size, axis=0)
    tensor_K = assembler.build_diffusion_operator(mesh, dr, diffusion=tensor)
    np.testing.assert_allclose(K.toarray(), tensor_K.toarray())
    np.testing.assert_allclose(K.sum(axis=1), 0, atol=1e-12)
    np.testing.assert_array_equal(M.toarray(), np.eye(mesh.size))
    center = np.ravel_multi_index((1,) * ndim, mesh.shape)
    np.testing.assert_allclose(K[center, center], 2 * ndim * 2 / dr**2)
    boundary_factor = 1 if operator is fw.AsymmetricDiffusion else 2
    np.testing.assert_allclose(K[0, 0], boundary_factor * ndim * 2 / dr**2)
    np.testing.assert_allclose(K[0, 1], -boundary_factor * 2 / dr**2)
    if operator is fw.AsymmetricDiffusion:
        np.testing.assert_allclose(K.toarray(), K.toarray().T)


@pytest.mark.parametrize('ndim', [2, 3])
def test_gradient_linear_field_including_boundaries(ndim):
    mesh = np.ones((3,) * ndim, dtype=np.int8)
    dr = 0.3
    coords = np.indices(mesh.shape).reshape(ndim, -1) * dr
    slopes = np.arange(1, ndim + 1)
    values = slopes @ coords + 5
    gradient = fw.FiniteDifferenceGradient()
    direct = gradient.build_gradient_operator(mesh, dr=dr)
    facade = fw.FiniteDifferenceDiscretization().build_gradient_operator(mesh, dr=dr)
    for axis, (op, delegated) in enumerate(zip(direct, facade)):
        np.testing.assert_allclose(op @ values, slopes[axis], atol=1e-12)
        np.testing.assert_array_equal(op.toarray(), delegated.toarray())


def test_holes_fibrosis_and_isolated_gradient():
    mesh = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]], dtype=np.int8)
    assembler = fw.FiniteDifferenceDiscretization()
    K, M = assembler.build_system_matrices(mesh)
    assert K.shape == M.shape == (5, 5)
    np.testing.assert_allclose(K.sum(axis=1), 0)
    np.testing.assert_array_equal(K.toarray()[2:], 0)
    for op in assembler.build_gradient_operator(mesh):
        assert op.shape == (5, 5)
        np.testing.assert_allclose(op @ np.ones(5), 0)
        np.testing.assert_array_equal(op.toarray()[2:], 0)


@pytest.mark.parametrize('operator', [fw.AsymmetricDiffusion, fw.IsotropicDiffusion])
def test_tissue_weights(operator):
    tissue = fw.CardiacTissueGrid((4, 4), dr=0.25)
    facade = fw.FiniteDifferenceDiscretization()
    facade.diffusion = operator()
    K, M = facade.compute_weights(tissue, D_model=2.)
    np.testing.assert_array_equal(M.toarray(), np.eye(K.shape[0]))
    assert M.format == K.format == "csr"
    direct = operator().build_diffusion_operator(
        tissue.mesh, tissue.dr,
        indexes=tissue.tissue_indexes[tissue.myo_indexes],
        diffusion=tissue.diffusion_tensor,
        connectivity=tissue.connectivity)
    np.testing.assert_allclose(K.toarray(), 2 * direct.toarray())


def test_default_grid_simulation_uses_facade():
    simulation = fw.CardiacSimulation(dt=0.01, t_max=0.02, backend='numba')
    simulation.cardiac_tissue = fw.CardiacTissueGrid((4, 4), dr=0.25)
    simulation.cardiac_model = fw.AlievPanfilov()
    simulation.run(prog_bar=False)
    assert type(simulation.spatial_discretization) is fw.FiniteDifferenceDiscretization
    assert np.all(np.isfinite(simulation.cardiac_model.u))


def test_asymmetric_anisotropic_quadratic_field():
    mesh = np.ones((5, 5), dtype=np.int8)
    dr = 0.2
    coords = np.indices(mesh.shape).reshape(2, -1) * dr
    diffusion = np.repeat(np.array([[[2., 0.3], [0.3, 1.]]]), mesh.size, axis=0)
    K = fw.AsymmetricDiffusion().build_diffusion_operator(mesh, dr, diffusion=diffusion)
    x, y = coords
    values = x*x + x*y + y*y
    # -div(D grad(u)) = -(2 Dxx + 2 Dxy + 2 Dyy).
    result = (K @ values).reshape(mesh.shape)
    np.testing.assert_allclose(result[1:-1, 1:-1], -6.6, atol=1e-12)
