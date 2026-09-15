import numpy as np

import finitewave as fw


def test_isotropic_discretization_builds_stiffness_and_mass_matrices():
    dr = 0.3
    diffusion = 0.5
    mesh = np.ones((3, 3), dtype=np.int8)
    tissue = fw.CardiacTissue(mesh=mesh, dr=dr)
    tissue.conductivity = diffusion

    stiffness, mass = fw.IsotropicDiscretization().compute_weights(tissue)

    assert stiffness.shape == (mesh.size, mesh.size)
    assert mass.shape == (mesh.size, mesh.size)
    np.testing.assert_allclose(stiffness.sum(axis=1).A1, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        stiffness.diagonal(), 4 * diffusion / dr**2, atol=1e-12
    )
    np.testing.assert_allclose(mass.diagonal(), 1.0)


def test_fibrotic_cell_stays_in_tissue_matrix_but_is_decoupled():
    mesh = np.ones((3, 3), dtype=np.int8)
    mesh[0, 0] = 2
    tissue = fw.CardiacTissue(mesh=mesh, dr=0.3)

    stiffness, mass = fw.IsotropicDiscretization().compute_weights(tissue)

    assert stiffness.shape == (mesh.size, mesh.size)
    assert mass.shape == (mesh.size, mesh.size)
    np.testing.assert_allclose(stiffness.getrow(0).toarray(), 0.0)
    np.testing.assert_allclose(stiffness.getcol(0).toarray(), 0.0)
    assert mass[0, 0] == 1.0
