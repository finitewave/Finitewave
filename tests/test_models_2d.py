import os
import shutil
import numpy as np
import pytest
import finitewave as fw


def prepare_model(model_class, curr_value, curr_dur, t_calc, t_prebeats):
    ni = 100
    nj = 3
    tissue = fw.CardiacTissue2D([ni, nj])

    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(fw.StimCurrentCoord2D(0, curr_value, curr_dur, 0, 2, 0, nj))
    stim_sequence.add_stim(fw.StimCurrentCoord2D(t_prebeats, curr_value, curr_dur, 0, 2, 0, nj))
    stim_sequence.add_stim(fw.StimCurrentCoord2D(2*t_prebeats, curr_value, curr_dur, 0, 2, 0, nj))
    stim_sequence.add_stim(fw.StimCurrentCoord2D(3*t_prebeats, curr_value, curr_dur, 0, 2, 0, nj))

    model = model_class()
    model.dt = 0.001
    model.dr = 0.1
    model.t_max = 3*t_prebeats + t_calc
    model.cardiac_tissue = tissue
    model.stim_sequence = stim_sequence

    return model

def run_model(model):
    tracker = fw.ActionPotential2DTracker()
    tracker.cell_ind = [50, 1]
    tracker.step = 1

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)
    model.tracker_sequence = seq

    model.run()
    return tracker.output

def calculate_apd(u, dt, threshold=0.1, beat_index=3):
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

@pytest.mark.aliev_panfilov
def test_aliev_panfilov_apd():
    model = prepare_model(fw.AlievPanfilov2D, curr_value=10, curr_dur=0.5, t_calc=80, t_prebeats=60)
    u = run_model(model)

    assert np.max(u) == pytest.approx(1.0, abs=0.02)
    assert np.min(u) == pytest.approx(0.0, abs=0.01)

    apd = calculate_apd(u, model.dt)
    assert 21 <= apd <= 23, f"Aliev-Panfilov APD90 is out of expected range {apd}"

@pytest.mark.luo_rudy91
def test_luo_rudy_apd():
    model = prepare_model(fw.LuoRudy912D, curr_value=10, curr_dur=1, t_calc=1000, t_prebeats=1000)
    u = run_model(model)

    assert np.max(u) == pytest.approx(40, abs=2)
    assert np.min(u) == pytest.approx(-84.5, abs=2)

    apd = calculate_apd(u, model.dt)
    assert 350 <= apd <= 400, f"Luo-Rudy APD90 is out of expected range {apd}"

# def test_tp06_apd():
#     model = prepare_model(fw.TP062D, volt_value=20, t_max=500)
#     u = run_model(model)

#     assert np.max(u) > 20
#     assert np.min(u) < -80

#     apd = calculate_apd(u, model.dt)
#     assert 350 <= apd <= 400