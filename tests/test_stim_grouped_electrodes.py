import numpy as np
import pytest

from finitewave import StimGroupedElectrodes
from finitewave.core.stimul.stim_type.stim_current import StimCurrent


def test_groups_preserve_time_coordinate_pairs_and_include_last_boundary():
    protocol = StimGroupedElectrodes([], np.empty((0, 2)), size=1)
    points, times = protocol.build_stim_groups(
        [2.0, 0.8, 0.2, 1.0], [[20, 0], [8, 0], [2, 0], [10, 0]])
    assert len(points) == len(times) == 3
    np.testing.assert_array_equal(times[0], [0.8, 0.2])
    np.testing.assert_array_equal(points[0], [[8, 0], [2, 0]])
    np.testing.assert_array_equal(times[-1], [2.0])
    np.testing.assert_array_equal(points[-1], [[20, 0]])


@pytest.mark.parametrize("times, step, expected", [
    ([1.0], 1.0, [1.0]),
    ([0.0, 0.5, 1.0], 0.5, [0.0, 0.5, 1.0]),
    ([0.2, 3.0], 5.0, [0.2]),
    ([-0.5, -1.0, 0.0], 0.5, [-1.0, -0.5, 0.0]),
    ([], 1.0, []),
])
def test_schedule_boundaries(times, step, expected):
    protocol = StimGroupedElectrodes(times, np.zeros((len(times), 2)), 1, step)
    protocol.add_stim(StimCurrent(10, 2, 0.1))
    assert [stim.t for stim in protocol.sequence] == expected


def test_template_copies_initialize_and_apply_to_each_electrode_group():
    from types import SimpleNamespace

    coords = np.array([[0., 0.], [1., 0.], [2., 0.]])
    template = StimCurrent(10, 2, 0.1)
    protocol = StimGroupedElectrodes([0.8, 0.2, 2.0], coords, 0.1)
    assert protocol.add_stim(template) is protocol
    first, second = protocol.sequence
    assert first is not second and first is not template
    assert template.t == 10 and template.stim_area is None
    assert first.t == 0.2 and second.t == 2.0
    assert first.duration == second.duration == 0.1

    def add_flat_values(array, indexes, value):
        array[indexes] += value
        return array

    simulation = SimpleNamespace(
        cardiac_tissue=SimpleNamespace(myo_coords=coords, myo_indexes=np.arange(3)),
        cardiac_model=SimpleNamespace(_rhs=np.zeros(3)),
        backend=SimpleNamespace(wrap_indexes=np.asarray, add_flat_values=add_flat_values),
        t=0.2, dt=0.01,
    )
    protocol.initialize(simulation)
    protocol.stimulate_next()
    np.testing.assert_array_equal(simulation.cardiac_model._rhs, [2, 2, 0])
    simulation.t = 2.0
    protocol.stimulate_next()
    np.testing.assert_array_equal(simulation.cardiac_model._rhs, [2, 2, 2])
    protocol.add_stim(template)
    assert len(protocol.sequence) == 4


@pytest.mark.parametrize("times, coords, step", [
    ([[0]], [[0, 0]], 1),
    ([np.nan], [[0, 0]], 1),
    ([np.inf], [[0, 0]], 1),
    ([0, 1], [[0, 0]], 1),
    ([0], [0, 0], 1),
    ([0], [[0, 0]], 0),
    ([0], [[0, 0]], -1),
    ([0], [[0, 0]], np.inf),
    ([0], [[0, 0]], np.nan),
])
def test_invalid_schedule(times, coords, step):
    protocol = StimGroupedElectrodes(times, coords, 1, step)
    with pytest.raises(ValueError):
        protocol.add_stim(StimCurrent(0, 2, 0.1))
    assert protocol.sequence == []
