from types import SimpleNamespace

import numpy as np
import pytest

from finitewave import ActionPotentialTracker, MultiVariableTracker
from finitewave.simulation.tracker.variable_tracker import VariableTracker


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


def make_simulation(backend, t_max=1.0):
    return SimpleNamespace(
        backend=backend, dt=0.1, t_max=t_max, t=0.0, iteration=0,
        cardiac_tissue=SimpleNamespace(mesh=np.ones((1, 2))),
        cardiac_model=SimpleNamespace(
            u=np.zeros(2), _u=backend.wrap_array(np.zeros(2)),
            init_u=0.0, tissue_indexes=np.arange(2)),
    )


def record(tracker, simulation, iterations):
    for iteration in iterations:
        simulation.iteration = iteration
        simulation.t = iteration * simulation.dt
        simulation.cardiac_model._u = simulation.backend.wrap_array(
            np.array([iteration, -iteration], dtype=float))
        tracker.track()


@pytest.mark.parametrize("tracker_class", [ActionPotentialTracker, VariableTracker,
                                           MultiVariableTracker])
@pytest.mark.parametrize("start,end,step,last", [
    (0.0, np.inf, 2, 10),  # Both initial and final samples.
    (0.0, np.inf, 3, 10),  # Final iteration is not sampled.
    (0.15, 0.85, 3, 10),  # Window is not aligned with sampling iterations.
    (0.4, 0.4, 2, 10),  # Exactly one sample retains its time axis.
    (2.0, np.inf, 2, 10),  # No samples.
    (0.0, np.inf, 2, 4),  # Stop before allocated storage is filled.
])
def test_output_matches_actual_samples(backend, tracker_class, start, end, step, last):
    simulation = make_simulation(backend)
    options = dict(node_inds=[0, 0], start_time=start, end_time=end, step=step)
    if tracker_class is VariableTracker:
        options["var_name"] = "u"
    elif tracker_class is MultiVariableTracker:
        options["var_list"] = ["u"]
    tracker = tracker_class(**options)
    tracker.initialize(simulation)
    record(tracker, simulation, range(last + 1))
    expected = [i for i in range(last + 1)
                if i % step == 0 and start <= i * simulation.dt <= end]
    values = tracker.output["u"] if tracker_class is MultiVariableTracker else tracker.output
    assert values.shape == (len(expected),)
    assert tracker.tracking_counter == len(expected)
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_allclose(tracker.tracking_times, np.array(expected) * simulation.dt)


def test_extended_run_and_reinitialization(backend):
    simulation = make_simulation(backend, t_max=0.2)
    tracker = ActionPotentialTracker(node_inds=[[0, 0], [0, 1]], step=2)
    tracker.initialize(simulation)
    record(tracker, simulation, range(3))
    simulation.t_max = 1.0
    record(tracker, simulation, range(3, 11))
    expected = np.arange(0, 11, 2)
    assert tracker.output.shape == (6, 2)
    assert len(tracker.tracking_times) == 6
    np.testing.assert_array_equal(tracker.output, np.column_stack([expected, -expected]))

    tracker.initialize(simulation)
    assert tracker.tracking_counter == 0
    assert len(tracker.tracking_times) == 0
    assert tracker.output.shape == (0, 2)
    record(tracker, simulation, range(3))
    np.testing.assert_array_equal(tracker.output, [[0, 0], [2, -2]])
    assert len(tracker.tracking_times) == 2
