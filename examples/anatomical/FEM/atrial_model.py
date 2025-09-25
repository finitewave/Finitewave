
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

import finitewave as fw


def load_mesh(path):
    coords = np.genfromtxt(path.joinpath("mesh.pts"), skip_header=1,
                           usecols=[0, 1, 2])
    coords /= 1000

    # print(coords.min(axis=0), coords.max(axis=0))
    elems = np.genfromtxt(path.joinpath("mesh.elem"), skip_header=1,
                          usecols=[1, 2, 3], dtype=int)
    return coords, elems


def prepace(path, stim_times, pacing_time, values, voltage_threshold):
    tissue = fw.CardiacTissueFDM((100, 10))

    for stim_time in stim_times:
        stim = fw.StimVoltageCoordFDM(stim_time, values, 0, 5, 0, 10)

    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(stim)

    prepacing_tracker = fw.PrepacingTrackerFDM()
    prepacing_tracker.path = path
    prepacing_tracker.cell_ind = [70, 5]
    prepacing_tracker.pacing_time = pacing_time
    prepacing_tracker.voltage_threshold = voltage_threshold

    tracker_sequence = fw.TrackerSequence()
    tracker_sequence.add_tracker(prepacing_tracker)

    model = fw.CourtemancheFDM()
    model.cardiac_tissue = tissue
    model.stim_sequence = stim_sequence
    model.tracker_sequence = tracker_sequence

    model.run()

    prepacing_tracker.write()


# create a tissue of size 400x400 with cardiomycytes:
path = Path(__file__).parents[2].joinpath("data", "atrial_mesh")
coords, elems = load_mesh(path)

# prepace(path, [0, 45], 1, [1, 1], 0)

state_loader = fw.StateLoader()
state_loader.path = path.joinpath("prepacing")

stim_coord = coords[len(coords) // 2]
stim_size = 1
stim = fw.StimVoltageElectrodesFEM(0, 1, stim_coord, stim_size)

tissue = fw.CardiacTissueFEM(coords, elems)
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(stim)

# create model object and set up parameters:
model = fw.CourtemancheFEM()
model.dt = 0.01
model.t_max = 40
# add the tissue and the stim parameters to the model object:
model.cardiac_tissue = tissue
model.stim_sequence = stim_sequence
# model.state_loader = state_loader

# run the model:
model.run(num_of_threads=None)

u = model.u

# show the potential map at the end of calculations:
faces = np.hstack([[3, *tri] for tri in elems])
mesh = pv.PolyData(coords, faces)
mesh.point_data["values"] = u
# plot
mesh.plot(cmap="RdBu_r", show_edges=False)
