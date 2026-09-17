import numpy as np

import finitewave as fw
from finitewave.simulation.fibrosis import DecouplingPattern


def test_line_decoupling_3d():
    tissue = fw.CardiacTissue(shape=(6, 5, 4), dr=0.25)
    coords = np.array([[2, 1, 1], [2, 2, 1], [2, 3, 1]])

    DecouplingPattern(coords=coords, axis=1).apply(tissue)

    assert tissue.connectivity.shape == tissue.mesh.shape + (3,)
    np.testing.assert_array_equal(
        tissue.connectivity[coords[:, 0], coords[:, 1], coords[:, 2], 1],
        0.0,
    )
    assert tissue.connectivity[0, 0, 0, 1] == 1.0


def test_region_decoupling_3d():
    tissue = fw.CardiacTissue(shape=(8, 8, 8), dr=0.25)
    region = [[2, 5], [3, 6], [1, 4]]

    DecouplingPattern(density=1.0, region=region, axis=2).apply(tissue)

    np.testing.assert_array_equal(
        tissue.connectivity[2:5, 3:6, 1:4, 2], 0.0
    )
    assert tissue.connectivity[0, 0, 0, 2] == 1.0
