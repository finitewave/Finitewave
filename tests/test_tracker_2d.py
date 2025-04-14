import os
import shutil
import numpy as np
import pytest
import finitewave as fw

@pytest.fixture
def tissue():
    ni = 50
    nj = 10
    tissue = fw.CardiacTissue2D([ni, nj])
    return tissue

@pytest.fixture
def model(tissue):
    stim = fw.StimSequence()
    stim.add_stim(fw.StimVoltageCoord2D(0, 1, 0, 5, 0, 10))
    
    model = fw.AlievPanfilov2D()
    model.dt = 0.01
    model.dr = 0.25
    model.t_max = 50
    model.cardiac_tissue = tissue
    model.stim_sequence = stim
    return model

def test_action_potential_2d_tracker(model):
    tracker = fw.ActionPotential2DTracker()
    tracker.cell_ind = [10, 5]
    tracker.step = 1

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)
    model.tracker_sequence = seq

    model.run()

    u = tracker.output

    # Check if the output is not empty
    assert u is not None
    assert len(u) > 0

    # Check if the Aliev-Panfilov model maximal amplitude is within expected range
    assert np.max(u) == pytest.approx(1.0, abs=0.02)

    threshold = 0.1
    up_idx = np.where((u[:-1] < threshold) & (u[1:] >= threshold))[0]
    down_idx = np.where((u[:-1] > threshold) & (u[1:] <= threshold))[0]

    assert len(up_idx) > 0, "Action potential upstroke not found"
    assert len(down_idx) > 0, "Action potential downstroke not found"

    # ap_start = up_idx[0]
    # ap_end = down_idx[down_idx > ap_start][0]

    # apd = (ap_end - ap_start) * model.dt
    # assert 25 <= apd <= 27, f"APD90 is out of expected range {apd}"

def test_animation_2d_tracker(model):
    tracker = fw.Animation2DTracker()
    tracker.variable_name = "u"
    tracker.dir_name = "test_frames"
    tracker.step = 100 # write every 100th step
    tracker.overwrite = True

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)
    model.tracker_sequence = seq

    model.run()

    # Check if the animation files are created
    assert os.path.exists(tracker.dir_name), "Output directory was not created."
    files = sorted(os.listdir(tracker.dir_name))
    expected_frames = (model.t_max/model.dt) // tracker.step
    assert len(files) == expected_frames, f"Expected {expected_frames} frames, got {len(files)}"

    # Check if the frames are not empty
    for fname in files:
        frame = np.load(os.path.join(tracker.dir_name, fname))
        assert np.any(frame > 0), f"Frame {fname} appears to be empty."

    shutil.rmtree(tracker.dir_name)

def test_activation_time_2d_tracker(model):
    tracker = fw.ActivationTime2DTracker()
    tracker.threshold = 0.5
    tracker.step = 1
    tracker.start_time = 0
    tracker.end_time = 100

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)
    model.tracker_sequence = seq

    model.stim_sequence.add_stim(fw.StimVoltageCoord2D(50, 1, 0, 5, 0, 10))
    model.t_max = 100

    model.run()

    ats = tracker.output

    # Check if the output is not empty
    assert ats is not None
    assert len(ats) > 0
    assert np.any(~np.isnan(ats)), "AT array is entirely NaN"

    # Check if the activation time values are within expected range
    assert ats[50//2, 10//2] == pytest.approx(3.5, abs=0.01)

def test_local_activation_time_2d_tracker(model):
    tracker = fw.LocalActivationTime2DTracker()
    tracker.threshold = 0.5
    tracker.step = 1
    tracker.start_time = 0
    tracker.end_time = 100

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)
    model.tracker_sequence = seq

    model.stim_sequence.add_stim(fw.StimVoltageCoord2D(50, 1, 0, 5, 0, 10))
    model.t_max = 100

    model.run()

    lats = tracker.output

    # Check if the output is not empty
    assert lats is not None
    assert len(lats) > 0
    assert np.any(~np.isnan(lats)), "LAT array is entirely NaN"

    # Values at the center cell should have two LAT values
    assert len(lats) == 2, "Every cell should have two LAT values"
    LAT1, LAT2 = lats[:, 50//2, 10//2]

    # Check if the LAT values are within expected range
    assert LAT1 < LAT2, "LAT values should be in ascending order"
    assert LAT1 == pytest.approx(3.5, abs=0.01)
    assert LAT2 == pytest.approx(53.68, abs=0.01)

def test_multi_variable_2d_tracker(model):
    tracker = fw.MultiVariable2DTracker()
    tracker.cell_ind = [10, 5]
    tracker.var_list = ["v"]

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)
    model.tracker_sequence = seq

    model.run()

    v = tracker.output["v"]

    # Check if the output is not empty
    assert v is not None
    assert len(v) > 0

    # Check if the Aliev-Panfilov model 'v' maximal amplitude is within expected range
    assert np.max(v) == pytest.approx(2, abs=0.1)

