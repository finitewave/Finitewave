
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


# path = Path("/Users/arstanbek/Projects/fibrosis/ElementalWave/data")
path = Path("/Users/arstanbek/Projects/fibrosis/Finitewave/examples/data/atrial_mesh")

coords, elems = load_mesh(path)
tissue = fw.CardiacTissueElements(coords, elems)
# tissue.mesh += (np.random.random(coords.shape[0]) < 0.2)

print(tissue.mesh)

# create model object and set up parameters
cardiac_model = fw.Courtemanche()
# Here, we increase g_Kur by a factor of 3 to better match physiological AP shape
# with a visible plateau and realistic repolarization.
# courtemanche.gkur_coeff *= 3
cardiac_model.gkur_coeff *= 0.5
cardiac_model.gto *= 0.5
cardiac_model.gcal *= 0.3

stim_coords = coords[1000:1001]
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageElectrodes(0, 1, stim_coords, 1))
# stim_sequence.add_stim(fw.StimVoltageCoord(45, 1, 0, size//2, 0, size))

# create model object and set up parameters:
simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.t_max = 50
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.AlievPanfilov()
simulation.stim_sequence = stim_sequence
# simulation.stencil = stencil

# run the model:
simulation.run()

u = simulation.cardiac_model.u

# show the potential map at the end of calculations:
faces = np.hstack([[3, *tri] for tri in elems])
mesh = pv.PolyData(coords, faces)
mesh.point_data["values"] = u
# plot
mesh.plot(cmap="RdBu_r", show_edges=False)
