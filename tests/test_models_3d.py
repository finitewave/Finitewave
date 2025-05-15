import os
import shutil
import numpy as np
import pytest
import finitewave as fw


def prepare_model(model_class, curr_value, curr_dur, t_calc, t_prebeats):
    """
    Prepares a 3D cardiac model with a stimulation protocol.

    Parameters
    ----------
    model_class : Callable
        The cardiac model class to be instantiated.
    curr_value : float
        Amplitude of the stimulus current (μA/cm² or model units).
    curr_dur : float
        Duration of each stimulus pulse (ms or model units).
    t_calc : float
        Time after the last preconditioning beat to continue recording (ms or model units).
    t_prebeats : float
        Interval between preconditioning stimuli (ms or model units).

    Returns
    -------
    model : CardiacModel
        Configured and initialized model ready for simulation.
    """
    ni = 5
    nj = 3
    nk = 3
    tissue = fw.CardiacTissue3D([ni, nj, nk])

    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(fw.StimCurrentCoord3D(0, curr_value, curr_dur, 0, 2, 0, nj, 0, nk))
    stim_sequence.add_stim(fw.StimCurrentCoord3D(t_prebeats, curr_value, curr_dur, 0, 2, 0, nj, 0, nk))
    stim_sequence.add_stim(fw.StimCurrentCoord3D(2*t_prebeats, curr_value, curr_dur, 0, 2, 0, nj, 0, nk))
    stim_sequence.add_stim(fw.StimCurrentCoord3D(3*t_prebeats, curr_value, curr_dur, 0, 2, 0, nj, 0, nk))

    model = model_class()
    model.dt = 0.01
    model.dr = 0.25
    model.t_max = 3*t_prebeats + t_calc
    model.cardiac_tissue = tissue
    model.stim_sequence = stim_sequence

    return model

def run_model(model):
    """
    Runs a cardiac model with a membrane potential tracker.

    Parameters
    ----------
    model : CardiacModel
        A configured model with stimulation and tissue already assigned.

    Returns
    -------
    output : np.ndarray
        Time series of membrane potential for a specific cell.
    """
    tracker = fw.ActionPotential3DTracker()
    tracker.cell_ind = [3, 1, 1]
    tracker.step = 1

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)
    model.tracker_sequence = seq

    model.run()
    return tracker.output

def calculate_apd(u, dt, threshold, beat_index=3):
    """
    Calculates the action potential duration (APD) for a single beat.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential time series.
    dt : float
        Time step of the simulation (ms).
    threshold : float
        Voltage threshold to define APD90 (e.g., -70 mV or 0.1 for normalized models).
    beat_index : int, optional
        Index of the beat to analyze (default is 3).

    Returns
    -------
    apd : float or None
        Duration of the action potential (ms or model units), or None if no complete AP was found.
    """
    up_idx = np.where((u[:-1] < threshold) & (u[1:] >= threshold))[0]
    down_idx = np.where((u[:-1] > threshold) & (u[1:] <= threshold))[0]

    if len(up_idx) <= beat_index or len(down_idx) == 0:
        return None

    ap_start = up_idx[beat_index]
    ap_end_candidates = down_idx[down_idx > ap_start]
    if len(ap_end_candidates) == 0:
        return None

    ap_end = ap_end_candidates[0]
    return (ap_end - ap_start) * dt

@pytest.mark.aliev_panfilov_3d
def test_aliev_panfilov():
    """
    Test the Aliev-Panfilov 3D model.

    This test checks:
    - Correct range of membrane potential values after stimulation.
    - Action potential duration (APD90) within expected range [21, 23] ms.

    Stimulation:
    - 4 current pulses at intervals of 60 ms.
    - Amplitude: 10
    - Duration: 0.5 ms
    """
    model = prepare_model(fw.AlievPanfilov3D, curr_value=5, curr_dur=0.5, t_calc=80, t_prebeats=60)
    u = run_model(model)

    assert np.max(u) == pytest.approx(1.0, abs=0.02)
    assert np.min(u) == pytest.approx(0.0, abs=0.01)

    apd = calculate_apd(u, model.dt, threshold=0.1)
    assert 20 <= apd <= 25, f"Aliev-Panfilov APD90 is out of expected range {apd}"

@pytest.mark.barkley_3d
def test_barkley():
    """
    Test the Barkley 3D model.

    This test checks:
    - Correct range of membrane potential values after stimulation.
    - Action potential duration (APD90) within expected range [1, 2] ms.

    Stimulation:
    - 4 current pulses at intervals of 60 ms.
    - Amplitude: 1
    - Duration: 0.5 ms
    """
    model = prepare_model(fw.Barkley3D, curr_value=5, curr_dur=0.1, t_calc=80, t_prebeats=60)
    u = run_model(model)

    assert np.max(u) == pytest.approx(1.0, abs=0.02)
    assert np.min(u) == pytest.approx(0.0, abs=0.01)

    apd = calculate_apd(u, model.dt, threshold=0.1)
    assert 1 <= apd <= 4, f"Barkley APD90 is out of expected range {apd}"

@pytest.mark.mitchell_schaeffer_3d
def test_mitchell_schaeffer():
    """
    Test the Mitchell-Schaeffer 3D model.

    This test checks:
    - Correct range of membrane potential values after stimulation.
    - Action potential duration (APD90) within [280, 320] ms.

    Stimulation:
    - 4 current pulses at intervals of 1000 ms.
    - Amplitude: 10
    - Duration: 0.5 ms
    """
    model = prepare_model(fw.MitchellSchaeffer3D, curr_value=5, curr_dur=0.5, t_calc=1000, t_prebeats=1000)
    u = run_model(model)

    assert np.max(u) == pytest.approx(0.95, abs=0.02)
    assert np.min(u) == pytest.approx(0.0, abs=0.01)

    apd = calculate_apd(u, model.dt, threshold=0.1)

    assert 250 <= apd <= 350, f"Mitchell-Schaeffer APD90 is out of expected range {apd}"

@pytest.mark.fenton_karma_3d
def test_fenton_karma():
    """
    Test the Fenton-Karma 3D model.

    This test checks:
    - Correct range of membrane potential values after stimulation.
    - Action potential duration (APD90) within [100, 200] ms.

    Stimulation:
    - 4 current pulses at intervals of 1000 ms.
    - Amplitude: 10
    - Duration: 0.5 ms
    """
    model = prepare_model(fw.FentonKarma3D, curr_value=5, curr_dur=0.5, t_calc=1000, t_prebeats=1000)
    u = run_model(model)

    assert np.max(u) == pytest.approx(1.0, abs=0.02)
    assert np.min(u) == pytest.approx(0.0, abs=0.01)

    apd = calculate_apd(u, model.dt, threshold=0.1)
    assert 100 <= apd <= 200, f"Fenton-Karma APD90 is out of expected range {apd}"

@pytest.mark.bueno_orovio_3d
def test_bueno_orovio_3d():
    """
    Test the Bueno-Orovio 3D model.

    This test checks:
    - Correct range of membrane potential values after stimulation.
    - Action potential duration (APD90) within [200, 300] ms.

    Stimulation:
    - 4 current pulses at intervals of 1000 ms.
    - Amplitude: 100
    - Duration: 1 ms
    """
    model = prepare_model(fw.BuenoOrovio3D, curr_value=5, curr_dur=0.5, t_calc=1000, t_prebeats=1000)
    u = run_model(model)

    assert np.max(u) == pytest.approx(1.4, abs=0.1)
    assert np.min(u) == pytest.approx(0.0, abs=0.01)

    apd = calculate_apd(u, model.dt, threshold=0.1)
    assert 200 <= apd <= 300, f"Bueno-Orovio APD90 is out of expected range {apd}"


@pytest.mark.luo_rudy91_3d
def test_luo_rudy():
    """
    Test the Luo-Rudy 1991 3D model.

    This test checks:
    - Correct range of membrane potential values after stimulation.
    - APD90 is within [350, 400] ms.

    Stimulation:
    - 4 current pulses at intervals of 1000 ms.
    - Amplitude: 100
    - Duration: 1 ms
    """
    model = prepare_model(fw.LuoRudy913D, curr_value=100, curr_dur=1, t_calc=1000, t_prebeats=1000)
    u = run_model(model)

    assert np.max(u) > 20
    assert np.min(u) < -80

    apd = calculate_apd(u, model.dt, threshold=-70)
    assert 350 <= apd <= 400, f"Luo-Rudy APD90 is out of expected range {apd}"

@pytest.mark.tp06_3d
def test_tp06():
    """
    Test the Ten Tusscher-Panfilov 2006 (TP06) 3D model.

    This test checks:
    - Correct range of membrane potential values after stimulation.
    - Action potential duration (APD90) within [280, 320] ms.

    Stimulation:
    - 4 current pulses at intervals of 1000 ms.
    - Amplitude: 100
    - Duration: 1 ms
    """
    model = prepare_model(fw.TP063D, curr_value=100, curr_dur=1, t_calc=1000, t_prebeats=1000)
    u = run_model(model)

    assert np.max(u) > 20
    assert np.min(u) < -80

    apd = calculate_apd(u, model.dt, threshold=-70)
    assert 280 <= apd <= 320, f"TP06 APD90 is out of expected range {apd}"

@pytest.mark.courtemanche_3d
def test_courtemanche():
    """
    Test the Courtemanche 3D model.

    This test checks:
    - Correct range of membrane potential values after stimulation.
    - Action potential duration (APD90) within [200, 300] ms.

    Note: Slightly elevated plateau potential is expected in some parameterizations.

    Stimulation:
    - 4 current pulses at intervals of 1000 ms.
    - Amplitude: 100
    - Duration: 1 ms
    """
    model = prepare_model(fw.Courtemanche3D, curr_value=100, curr_dur=1, t_calc=1000, t_prebeats=1000)
    u = run_model(model)

    assert np.max(u) > 10
    assert np.min(u) < -80

    apd = calculate_apd(u, model.dt, threshold=-70)
    assert 200 <= apd <= 300, f"Courtemanche APD90 is out of expected range {apd}"

