import os
import shutil
import numpy as np
import pytest
import finitewave as fw


def prepare_model(model_class, curr_value, curr_dur, t_calc, t_prebeats, dt, dr):
    """
    Prepares a cardiac model with a stimulation protocol.

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
    dt : float
        Time step for the simulation (ms or model units).
    dr : float
        Spatial step for the simulation (mm or model units).
        
    Returns
    -------
    model : CardiacModel
        Configured and initialized model ready for simulation.
    """
    ni = 50
    nj = 9
    tissue = fw.CardiacTissue([ni, nj])

    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(fw.StimCurrentCoord(0, curr_value, curr_dur, 0, 2, 0, nj))
    stim_sequence.add_stim(fw.StimCurrentCoord(t_prebeats, curr_value, curr_dur, 0, 2, 0, nj))
    stim_sequence.add_stim(fw.StimCurrentCoord(2*t_prebeats, curr_value, curr_dur, 0, 2, 0, nj))
    stim_sequence.add_stim(fw.StimCurrentCoord(3*t_prebeats, curr_value, curr_dur, 0, 2, 0, nj))

    model = model_class()
    model.dt = dt
    model.dr = dr
    model.t_max = 3*t_prebeats + t_calc
    model.cardiac_tissue = tissue
    model.stim_sequence = stim_sequence

    return model

def run_model(model, activation_time_start, activation_time_threshold):
    """
    Runs a cardiac model with a membrane potential tracker.

    Parameters
    ----------
    model : CardiacModel
        A configured model with stimulation and tissue already assigned.
    activation_time_start : float
        The start time for activation time tracking.
    activation_time_threshold : float
        The threshold for detecting activation.

    Returns
    -------
    output : np.ndarray
        Time series of membrane potential for a specific cell.
    """
    tracker = fw.LocalActivationTimeTracker()
    tracker.start_time = activation_time_start
    tracker.threshold = activation_time_threshold
    tracker.step = 1

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)
    model.tracker_sequence = seq

    model.run()

    return tracker.output[-1]

def calculate_wave_speed(activation_times, dr):
    """Calculates the wave speed from activation times.
    
    Parameters
    ----------
    activation_times : np.ndarray
        2D array of activation times for each cell in the tissue.
    dr : float
        Spatial step size of the simulation.
    
    Returns
    -------
    speed : float
        Estimated wave speed in mm/ms or model units.
    """
    time_diffs = np.diff(activation_times, axis=0)

    assert np.all(np.isfinite(time_diffs)), "Activation times contain NaN or inf"
    
    avg_time_diff = np.nanmean(time_diffs)
    assert avg_time_diff > 0, "Activation times must increase along x"

    return dr/avg_time_diff

@pytest.mark.propagation_aliev_panfilov_2d
def test_propagation_aliev_panfilov_2d():
    model = prepare_model(fw.AlievPanfilov, curr_value=5, curr_dur=1.0, t_calc=100, t_prebeats=1000, dt=0.01, dr=0.25)
    activation_time = run_model(model, activation_time_start=0, activation_time_threshold=0.5)

    # 3:-3 excludes stimulated nodes and boundary effects
    # 1:-1 excludes transverse boundaries
    speed = calculate_wave_speed(activation_time[3:-3, 1:-1], model.dr) 
    assert speed == pytest.approx(1.6, abs=0.01), f"Calculated wave speed {speed} is out of expected range"

@pytest.mark.propagation_barkley_2d
def test_propagation_barkley_2d():
    model = prepare_model(fw.Barkley, curr_value=5, curr_dur=1.0, t_calc=100, t_prebeats=1000, dt=0.01, dr=0.25)
    activation_time = run_model(model, activation_time_start=0, activation_time_threshold=0.5)

    speed = calculate_wave_speed(activation_time[3:-3, 1:-1], model.dr) 
    assert speed == pytest.approx(1.64, abs=0.01), f"Calculated wave speed {speed} is out of expected range"

@pytest.mark.propagation_mitchell_schaeffer_2d
def test_propagation_mitchell_schaeffer_2d():
    model = prepare_model(fw.MitchellSchaeffer, curr_value=5, curr_dur=1.0, t_calc=100, t_prebeats=1000, dt=0.01, dr=0.25)
    activation_time = run_model(model, activation_time_start=0, activation_time_threshold=0.5)

    speed = calculate_wave_speed(activation_time[3:-3, 1:-1], model.dr) 
    assert speed == pytest.approx(0.58, abs=0.01), f"Calculated wave speed {speed} is out of expected range"

@pytest.mark.propagation_fenton_karma_2d
def test_propagation_fenton_karma_2d():
    model = prepare_model(fw.FentonKarma, curr_value=5, curr_dur=1.0, t_calc=100, t_prebeats=1000, dt=0.01, dr=0.25)
    activation_time = run_model(model, activation_time_start=0, activation_time_threshold=0.5)

    speed = calculate_wave_speed(activation_time[3:-3, 1:-1], model.dr) 
    assert speed == pytest.approx(0.58, abs=0.01), f"Calculated wave speed {speed} is out of expected range"

@pytest.mark.propagation_bueno_orovio_2d
def test_propagation_bueno_orovio_2d():
    model = prepare_model(fw.BuenoOrovio, curr_value=5, curr_dur=1.0, t_calc=100, t_prebeats=1000, dt=0.01, dr=0.25)
    activation_time = run_model(model, activation_time_start=0, activation_time_threshold=0.5)

    speed = calculate_wave_speed(activation_time[3:-3, 1:-1], model.dr) 
    assert speed == pytest.approx(0.64, abs=0.01), f"Calculated wave speed {speed} is out of expected range"

@pytest.mark.propagation_luo_rudy91_2d
def test_propagation_luo_rudy91_2d():
    model = prepare_model(fw.LuoRudy91, curr_value=100, curr_dur=1.5, t_calc=100, t_prebeats=1000, dt=0.01, dr=0.25)
    activation_time = run_model(model, activation_time_start=0, activation_time_threshold=-60)

    speed = calculate_wave_speed(activation_time[3:-3, 1:-1], model.dr) 
    assert speed == pytest.approx(0.56, abs=0.01), f"Calculated wave speed {speed} is out of expected range"

@pytest.mark.propagation_tp06_2d
def test_propagation_tp06_2d():
    model = prepare_model(fw.TenTusscherPanfilov2006, curr_value=100, curr_dur=1.5, t_calc=100, t_prebeats=1000, dt=0.01, dr=0.25)
    activation_time = run_model(model, activation_time_start=0, activation_time_threshold=-60)

    speed = calculate_wave_speed(activation_time[3:-3, 1:-1], model.dr) 
    assert speed == pytest.approx(0.72, abs=0.01), f"Calculated wave speed {speed} is out of expected range"

@pytest.mark.propagation_courtemanche_2d
def test_propagation_courtemanche_2d():
    model = prepare_model(fw.Courtemanche, curr_value=100, curr_dur=1.5, t_calc=100, t_prebeats=1000, dt=0.01, dr=0.25)
    activation_time = run_model(model, activation_time_start=0, activation_time_threshold=-60)

    speed = calculate_wave_speed(activation_time[3:-3, 1:-1], model.dr) 
    assert speed == pytest.approx(0.58, abs=0.01), f"Calculated wave speed {speed} is out of expected range"
