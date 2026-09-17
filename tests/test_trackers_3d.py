import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pytest
import finitewave as fw
from finitewave.simulation.tracker.ecg_tracker import ECGTracker
from finitewave.simulation.tracker.frame_tracker import FrameTracker
from finitewave.simulation.tracker.local_activation_time_tracker import (
    LocalActivationTimeTracker,
)
from finitewave.simulation.tracker.multi_variable_tracker import MultiVariableTracker
from finitewave.simulation.tracker.period_tracker import PeriodTracker

@pytest.fixture
def cable_model():
    ni = 12
    nj = 3
    nk = 3
    tissue = fw.CardiacTissue(shape=(ni, nj, nk), dr=0.25)

    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(fw.StimCurrentCoord(0, 5, 0.5, 0, 5, 0, nj, 0, nk))
    
    simulation = fw.CardiacSimulation(dt=0.01, t_max=3)
    simulation.cardiac_tissue = tissue
    simulation.cardiac_model = fw.AlievPanfilov()
    simulation.stim_sequence = stim_sequence
    return simulation

@pytest.fixture
def spiral_model():
    ni = 100
    nj = 100
    nk = 3
    tissue = fw.CardiacTissue(shape=(ni, nj, nk), dr=0.25)

    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(fw.StimVoltageCoord(0, 1, 0, ni, 0, 3, 0, nk))
    stim_sequence.add_stim(fw.StimVoltageCoord(5, 1, 0, ni//2, 0, nj, 0, nk))
    
    simulation = fw.CardiacSimulation(dt=0.01, t_max=20)
    simulation.cardiac_tissue = tissue
    simulation.cardiac_model = fw.Barkley()
    simulation.stim_sequence = stim_sequence
    return simulation

@pytest.fixture
def planar_model():
    ni = 50
    nj = 5
    nk = 3
    tissue = fw.CardiacTissue(shape=(ni, nj, nk), dr=0.25)

    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(fw.StimCurrentCoord(5, 5, 0.5, 0, 5, 0, nj, 0, nk))
    
    simulation = fw.CardiacSimulation(dt=0.0015, t_max=15)
    simulation.cardiac_tissue = tissue
    simulation.cardiac_model = fw.AlievPanfilov()
    simulation.stim_sequence = stim_sequence
    return simulation

@pytest.mark.action_potential_3d_tracker
def test_action_potential_tracker(cable_model):
    tracker = fw.ActionPotentialTracker()
    tracker.node_inds = [10, 1, 1]
    tracker.step = 1

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)
    cable_model.tracker_sequence = seq

    cable_model.t_max = 30
    cable_model.run(prog_bar=False)

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

    ap_start = up_idx[0]
    ap_end = down_idx[down_idx > ap_start][0]

    apd = (ap_end - ap_start) * cable_model.dt
    # without prebeats:
    assert 20 <= apd <= 30, f"APD90 is out of expected range {apd}"

@pytest.mark.animation_3d_tracker
def test_frame_3d_tracker(spiral_model):
    tracker = FrameTracker()
    tracker.var_name = "u"
    tracker.dir_name = "test_frames"
    tracker.step = 100 # write every 100th step
    tracker.overwrite = True

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)
    spiral_model.tracker_sequence = seq

    spiral_model.run(prog_bar=False)

    # Check if the animation files are created
    assert os.path.exists(tracker.dir_name), "Output directory was not created."
    files = sorted(os.listdir(tracker.dir_name))
    expected_frames = (spiral_model.t_max/spiral_model.dt) // tracker.step
    assert len(files) == expected_frames, f"Expected {expected_frames} frames, got {len(files)}"

    # Check if the frames are not empty
    for fname in files:
        frame = np.load(os.path.join(tracker.dir_name, fname))
        assert np.any(frame > 0), f"Frame {fname} appears to be empty." 

    shutil.rmtree(tracker.dir_name)

@pytest.mark.activation_time_3d_tracker
def test_activation_time_3d_tracker(cable_model):
    # TODO:
    # Edge cases: start time - end time, values rewriting (should not work) 
    tracker = fw.ActivationTimeTracker()
    tracker.threshold = 0.5
    tracker.step = 1
    tracker.start_time = 0

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)
    cable_model.tracker_sequence = seq

    cable_model.run(prog_bar=False)

    ats = tracker.output

    # Check if the output is not empty
    assert ats is not None
    assert len(ats) > 0
    assert np.any(~np.isnan(ats)), "AT array is entirely NaN"

    # Check if the wavefront speed (distance/activation time) value is within expected range
    speed = 5*cable_model.cardiac_tissue.dr/ats[10, 1, 1] # 5 - number of nodes on the way
    assert 1.5 <= speed <= 2, f"Wavefront speed is out of expected range {speed}" 

@pytest.mark.local_activation_time_3d_tracker
def test_local_activation_time_3d_tracker(cable_model):
    tracker = LocalActivationTimeTracker()
    tracker.threshold = 0.5
    tracker.step = 1
    tracker.start_time = 0

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)
    cable_model.tracker_sequence = seq

    cable_model.stim_sequence.add_stim(fw.StimVoltageCoord(45, 1, 0, 5, 0, 10, 0, 3))
    
    cable_model.t_max = 50
    cable_model.run(prog_bar=False)

    lats = tracker.output

    # Check if the output is not empty
    assert lats is not None
    assert len(lats) > 0
    assert np.any(~np.isnan(lats)), "LAT array is entirely NaN"

    # Values at the center cell should have two LAT values
    assert len(lats) == 2, "Every cell should have two LAT values"
    LAT1, LAT2 = lats[:, 10, 1, 1]

    # Check if the wavefront speed (distance/activation time)  values are within expected range
    assert LAT1 < LAT2, "LAT values should be in ascending order"
    speed_1 = 5*cable_model.cardiac_tissue.dr/LAT1 # 5 - number of nodes on the way
    speed_2 = 5*cable_model.cardiac_tissue.dr/(LAT2 - 45) # 45 - second wave start time
    assert 1.5 <= speed_1 <= 2, f"Wavefront speed for the first wave is out of expected range {speed_1}" 
    assert 1.5 <= speed_2 <= 2, f"Wavefront speed for the second wave is out of expected range {speed_2}"

@pytest.mark.activation_time_3d_tracker
def test_multi_variable_3d_tracker(cable_model):
    tracker = MultiVariableTracker()
    tracker.node_inds = [10, 1, 1]
    tracker.var_list = ["v"]

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)
    cable_model.tracker_sequence = seq
    
    cable_model.t_max = 30
    cable_model.run(prog_bar=False)

    v = tracker.output["v"]

    # Check if the output is not empty
    assert v is not None
    assert len(v) > 0

    # Check if the Aliev-Panfilov model 'v' maximal amplitude is within expected range
    assert np.max(v) == pytest.approx(2, abs=0.15)

@pytest.mark.spiral_wave_core_3d_tracker
@pytest.mark.skip(reason="SpiralWaveCoreTracker is not available in the current tracker API")
def test_spiral_wave_core_3d_tracker(spiral_model):
    pass

@pytest.mark.spiral_wave_period_3d_tracker
def test_spiral_wave_period_3d_tracker(spiral_model):
    tracker = PeriodTracker()
    # Here we create an int array of detectors as a list of positions in which we want to calculate the period.
    positions = np.array([[80, 80, 1], [20, 70, 1], [40, 10, 1], [25, 90, 1]])
    tracker.node_inds = positions
    tracker.threshold = 0.5
    tracker.start_time = 10
    tracker.step = 10

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)
    spiral_model.tracker_sequence = seq

    spiral_model.run(prog_bar=False)

    periods = tracker.output

    period_mean = np.mean(np.array([np.mean(x) if len(x) > 0 else np.nan for x in periods]))

    # Check if the output is not empty
    assert periods is not None
    assert len(periods) > 0

    # Check if the spiral wave period is within expected range
    assert period_mean == pytest.approx(3.5, abs=0.2)

@pytest.mark.ecg_3d_tracker
def test_ecg_3d_tracker(planar_model):
    tracker = ECGTracker()
    tracker.start_time = 0
    tracker.step = 10
    tracker.measure_coords = np.array([[25, 2, 1]])

    seq = fw.TrackerSequence()
    seq.add_tracker(tracker)

    planar_model.tracker_sequence = seq

    planar_model.run(prog_bar=False)

    ecg = tracker.output.T[0]

    assert ecg.max() > 0.001
    assert ecg.min() < -0.001
    assert np.argmax(ecg) > 100  # Check if the peak occurs not at the beginning

# def test_animation_slice_3d_tracker():
    
#     class MockModel:
#         def __init__(self):
#             self.V = np.random.rand(5, 5, 5)
#             self.cardiac_tissue = type("Tissue", (), {})()
#             self.cardiac_tissue.mesh = np.ones((5, 5, 5), dtype=np.int8)

#     tracker = fw.AnimationSliceTracker()
#     tracker.variable_name = 'V'
#     tracker.slice_z = 2  # only one of slice_x, slice_y, slice_z must be set
#     tracker.dir_name = "test_frames"
#     tracker.file_name = "test_animation"

#     with TemporaryDirectory() as tmpdir:
#         tracker.path = tmpdir
#         model = MockModel()
#         tracker.initialize(model)

#         for _ in range(3):
#             tracker._track()

#         output_dir = Path(tmpdir) / "test_frames"
#         files = sorted(output_dir.glob("*.npy"))
#         assert len(files) == 3, "Should create exactly 3 frame files"

#         for file in files:
#             frame = np.load(file)
#             assert frame.shape == (5, 5), "Each frame should have shape (5, 5)"
#             assert frame.dtype == np.float32 or frame.dtype == np.float64

# def test_period_animation_3d_tracker(spiral_model):
#     tracker = fw.PeriodAnimationTracker()
#     tracker.dir_name = "test_frames"
#     tracker.threshold = 0.5
#     tracker.step = 100  # write every 100th step
#     tracker.overwrite = True

#     seq = fw.TrackerSequence()
#     seq.add_tracker(tracker)
#     spiral_model.tracker_sequence = seq

#     spiral_model.run()

#     # Check if the animation files are created
#     assert os.path.exists(tracker.dir_name), "Output directory was not created."
#     files = sorted(os.listdir(tracker.dir_name))
#     expected_frames = (spiral_model.t_max/spiral_model.dt) // tracker.step
#     assert len(files) == expected_frames, f"Expected {expected_frames} frames, got {len(files)}"

#     # Check if the frames are not empty
#     frame = np.load(os.path.join(tracker.dir_name, files[-1]))
#     assert np.any(frame > 0), f"Frame {frame} appears to be empty."

#     shutil.rmtree(tracker.dir_name)

