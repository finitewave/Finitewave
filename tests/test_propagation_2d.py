from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest

import finitewave as fw
from finitewave.simulation.tracker.local_activation_time_tracker import (
    LocalActivationTimeTracker,
)


@dataclass(frozen=True)
class PropagationCase:
    name: str
    model_class: type
    current: float
    duration: float
    t_max: float
    threshold: float
    speed_range: Tuple[float, float]
    speed_units: str
    validation: str


# Reference ranges check that dimensionless phenomenological models retain a
# stable propagation regime. They are not claims about biological CV! 
# Physiological ranges assume dr in mm, time in ms, and D in mm^2/ms.
PROPAGATION_CASES = (
    PropagationCase(
        "aliev_panfilov", fw.AlievPanfilov, 5, 0.5, 20, 0.5,
        (1.45, 1.70), "model units", "numerical reference",
    ),
    PropagationCase(
        "barkley", fw.Barkley, 5, 0.1, 20, 0.5,
        (3.8, 4.6), "model units", "numerical reference",
    ),
    PropagationCase(
        "mitchell_schaeffer", fw.MitchellSchaeffer, 5, 0.5, 50, 0.5,
        (0.50, 0.70), "model units", "numerical reference",
    ),
    PropagationCase(
        "fenton_karma", fw.FentonKarma, 5, 0.5, 50, 0.5,
        (0.50, 0.70), "model units", "numerical reference",
    ),
    PropagationCase(
        "bueno_orovio", fw.BuenoOrovio, 5, 0.5, 50, 0.5,
        (0.50, 0.90), "mm/ms", "physiological plausibility",
    ),
    PropagationCase(
        "luo_rudy91", fw.LuoRudy91, 100, 1.0, 50, -60,
        (0.45, 0.75), "mm/ms", "physiological plausibility",
    ),
    PropagationCase(
        "tp06", fw.TenTusscherPanfilov2006, 100, 1.5, 50, -60,
        (0.50, 0.90), "mm/ms", "physiological plausibility",
    ),
    PropagationCase(
        "courtemanche", fw.Courtemanche, 100, 1.5, 50, -60,
        (0.40, 0.80), "mm/ms", "physiological plausibility",
    ),
)


@dataclass(frozen=True)
class PropagationResult:
    case: PropagationCase
    activation_times: np.ndarray
    speed: float


def prepare_simulation(case, dt=0.01, dr=0.25):
    """Create the common plane-wave propagation experiment."""
    ni = 30
    nj = 3
    tissue = fw.CardiacTissue(shape=(ni, nj), dr=dr)

    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(
        fw.StimCurrentCoord(0, case.current, case.duration, 0, 2, 0, nj)
    )

    simulation = fw.CardiacSimulation(dt=dt, t_max=case.t_max)
    simulation.cardiac_tissue = tissue
    simulation.cardiac_model = case.model_class()
    simulation.stim_sequence = stim_sequence
    return simulation


def run_model(simulation, threshold):
    """Run a simulation and return its first local-activation-time map."""
    tracker = LocalActivationTimeTracker(
        start_time=0,
        threshold=threshold,
        step=1,
    )
    tracker_sequence = fw.TrackerSequence()
    tracker_sequence.add_tracker(tracker)
    simulation.tracker_sequence = tracker_sequence

    simulation.run(prog_bar=False)
    return tracker.output[-1]


def calculate_wave_speed(activation_times, dr):
    """Calculate mean plane-wave speed from adjacent activation times."""
    mean_time_difference = np.mean(np.diff(activation_times, axis=0))
    return dr / mean_time_difference


def propagation_parameter(case):
    marker = getattr(pytest.mark, f"propagation_{case.name}_2d")
    return pytest.param(case, id=case.name, marks=marker)


@pytest.fixture(
    scope="module",
    params=[propagation_parameter(case) for case in PROPAGATION_CASES],
)
def propagation_result(request):
    case = request.param
    simulation = prepare_simulation(case)
    activation_map = run_model(simulation, case.threshold)

    # Exclude the stimulated end, the far boundary, and transverse boundaries.
    activation_times = activation_map[3:-3, 1:-1]
    speed = calculate_wave_speed(
        activation_times,
        simulation.cardiac_tissue.dr,
    )
    return PropagationResult(case, activation_times, speed)


def test_wave_propagates_through_measurement_region(propagation_result):
    activation_times = propagation_result.activation_times

    assert activation_times.size > 0
    assert np.all(np.isfinite(activation_times))
    assert np.all(activation_times >= 0), "The wave did not activate every measured node"

    activation_delays = np.diff(activation_times, axis=0)
    assert np.all(activation_delays > 0), (
        "Activation times must increase along the propagation direction"
    )


def test_wave_speed_is_in_expected_range(propagation_result):
    case = propagation_result.case
    speed = propagation_result.speed
    lower, upper = case.speed_range

    assert lower <= speed <= upper, (
        f"Calculated wave speed {speed:.6g} {case.speed_units} is outside "
        f"the expected {case.validation} range [{lower}, {upper}]"
    )
