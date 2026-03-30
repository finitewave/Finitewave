
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.spatial import cKDTree

import finitewave as fw


# path = Path("/Users/arstanbek/Projects/fibrosis/ElementalWave/data")
path = Path("/Users/arstanbekokenov/Projects/Finitewave/examples/data/atrial_mesh")
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

# coords, elems = load_mesh(path)
tissue = fw.CardiacTissueElements(coords, elems, "Triangle")
tissue.conductivity = conductivity
tissue.fibers = fibers

stim_indexes = np.random.choice(coords.shape[0], size=10, replace=False)

lat_tracker = fw.LocalActivationTimeTracker()
lat_tracker.threshold = 0.5
lat_tracker.step = 10

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(lat_tracker)

cardiac_model = fw.AlievPanfilov()
# cardiac_model.prepacing()

# create model object and set up parameters:
simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.t_max = 20
simulation.cardiac_tissue = tissue
simulation.cardiac_model = cardiac_model
simulation.tracker_sequence = tracker_sequence

u = np.load(path / "u.npy")
v = np.load(path / "v.npy")

simulation.initialize()
simulation.cardiac_model.u = u
simulation.cardiac_model.v = v
# run the model:
simulation.run(num_of_threads=6, initialize=False)

u = simulation.cardiac_model.u
v = simulation.cardiac_model.v

lat_map = lat_tracker.activation_map(0, simulation.t_max)

# np.save(path / "lat_map_100.npy", lat_tracker.output)

faces = np.hstack([[3, *tri] for tri in elems])
mesh = pv.PolyData(coords, faces)
# pickable plot
plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=False, scalars=lat_map, cmap="magma")
plotter.show()
