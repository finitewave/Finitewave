from types import SimpleNamespace

import numpy as np

import finitewave as fw


class IdentityBackend:
    @staticmethod
    def wrap_indexes(indexes):
        return indexes


def selected_coordinates(stimulus, tissue):
    simulation = SimpleNamespace(
        cardiac_tissue=tissue,
        backend=IdentityBackend(),
    )
    stimulus.initialize(simulation)
    tissue_coordinates = tissue.coords[tissue.tissue_indexes]
    return tissue_coordinates[np.asarray(stimulus.stim_indexes)]


def test_coordinate_stimulus_uses_half_open_intervals_2d():
    tissue = fw.CardiacTissue(shape=(4, 5), dr=1.0)
    stimulus = fw.StimCurrentCoord(0, 1, 1, 1, 3, 2, 5)

    coordinates = selected_coordinates(stimulus, tissue)

    expected = np.array(
        [
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 2],
            [2, 3],
            [2, 4],
        ]
    )
    np.testing.assert_array_equal(coordinates, expected)


def test_adjacent_coordinate_stimuli_do_not_overlap():
    tissue = fw.CardiacTissue(shape=(4, 3), dr=1.0)
    left = fw.StimVoltageCoord(0, 1, 0, 2, 0, 3)
    right = fw.StimVoltageCoord(0, 1, 2, 4, 0, 3)

    left_coordinates = selected_coordinates(left, tissue)
    right_coordinates = selected_coordinates(right, tissue)

    left_indexes = {tuple(coord) for coord in left_coordinates}
    right_indexes = {tuple(coord) for coord in right_coordinates}
    assert left_indexes.isdisjoint(right_indexes)
    assert left_indexes | right_indexes == {
        (i, j) for i in range(4) for j in range(3)
    }
