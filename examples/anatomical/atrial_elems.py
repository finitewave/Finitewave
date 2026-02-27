
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
path = Path("/Users/arstanbekokenov/Projects/Finitewave/examples/data/atrial_mesh")

import pyvista as pv
vtk_mesh = pv.read(path / "Mesh_10954794.vtk")

iir = vtk_mesh.point_data['IIR']
iir = (1.22 - iir) / (1.22 - 1.0)
iir[iir < 0] = 0
iir[iir > 1] = 1
conductivity = 0.2 + 0.8 * iir

coords = vtk_mesh.points / 1000
coords /= 2
elems = vtk_mesh.faces.reshape(-1, 4)[:, 1:4]
fibers = vtk_mesh.cell_data['fiber_endo']
conductivity = conductivity[elems].mean(axis=1)

# print(coords.min(axis=0), coords.max(axis=0))

faces = np.hstack([[3, *tri] for tri in elems])
mesh = pv.PolyData(coords, faces)
geodesic = mesh.geodesic(19600, 72902)

# coords, elems = load_mesh(path)
tissue = fw.CardiacTissueElements(coords, elems, "Triangle")
# tissue.mesh_elems += (np.random.random(elems.shape[0]) < 0.2).astype(tissue.mesh_elems.dtype)
# tissue.mesh += (np.random.random(coords.shape[0]) < 0.1).astype(tissue.mesh.dtype)
tissue.conductivity = conductivity
tissue.fibers = fibers
# tissue.mesh += (np.random.random(coords.shape[0]) < 0.2)

# print(tissue.mesh.shape)

# create model object and set up parameters
cardiac_model = fw.Courtemanche()
# Here, we increase g_Kur by a factor of 3 to better match physiological AP shape
# with a visible plateau and realistic repolarization.
# courtemanche.gkur_coeff *= 3
cardiac_model.gkur_coeff *= 0.5
cardiac_model.gto *= 0.5
cardiac_model.gcal *= 0.3

stim_indexes = np.random.choice(coords.shape[0], size=10, replace=False)
stim_coords = coords[stim_indexes, :]
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimCurrentElectrodes(0, 30, 0.1, coords[19600:19601], 1))
stim_sequence.add_stim(fw.StimCurrentElectrodes(55, 15, 0.1, geodesic.points, 1))
stim_sequence.add_stim(fw.StimCurrentElectrodes(100, 15, 0.1, coords[72902:72903], 1))

# create model object and set up parameters:
simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.t_max = 10
simulation.state_loader = fw.StateLoader(path)
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.AlievPanfilov()
# simulation.stim_sequence = stim_sequence
# simulation.stencil = stencil

# run the model:
simulation.run(num_of_threads=6)

u = simulation.cardiac_model.u
v = simulation.cardiac_model.v
# np.save(path / "u.npy", u)
# np.save(path / "v.npy", v)

# # show the potential map at the end of calculations:
faces = np.hstack([[3, *tri] for tri in elems])
mesh = pv.PolyData(coords, faces)
mesh.point_data["values"] = u
# mesh.plot(cmap="RdBu_r", show_edges=True)

def callback(point):
    # Get closest point ID
    point_id = mesh.find_closest_point(point)
    print("Picked point ID:", point_id)

# pickable plot
plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=False, scalars=u, cmap="plasma")
# plotter.add_mesh(geodesic, color="red", line_width=3)
plotter.enable_point_picking(callback=callback, show_point=True)
plotter.show()
