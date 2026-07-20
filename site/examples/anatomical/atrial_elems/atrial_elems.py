
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.spatial import cKDTree

import finitewave as fw


def load_mesh(path):
    coords = np.genfromtxt(path.joinpath("mesh.pts"), skip_header=1,
                           usecols=[0, 1, 2])
    coords /= 1000

    # print(coords.min(axis=0), coords.max(axis=0))
    elems = np.genfromtxt(path.joinpath("mesh.elem"), skip_header=1,
                          usecols=[1, 2, 3], dtype=int)
    return coords, elems


# path = Path("/Users/arstanbek/Projects/fibrosis/ElementalWave/data")
path = Path("/Users/arstanbekokenov/Projects/Finitewave/examples/data/atrial_mesh")
vtk_mesh = pv.read(path / "Mesh_10954794.vtk")

iir = vtk_mesh.point_data['IIR']
iir = (1.22 - iir) / (1.22 - 1.0)
iir[iir < 0] = 0
iir[iir > 1] = 1
conductivity = 0.2 + 0.8 * iir

coords = vtk_mesh.points / 1000
coords /= 3
elems = vtk_mesh.faces.reshape(-1, 4)[:, 1:4]
fibers = vtk_mesh.cell_data['fiber_endo']
conductivity = conductivity[elems].mean(axis=1)

# print(coords.min(axis=0), coords.max(axis=0))

faces = np.hstack([[3, *tri] for tri in elems])
mesh = pv.PolyData(coords, faces)
geodesic = mesh.geodesic(19600, 72902)

# coords, elems = load_mesh(path)
tissue = fw.CardiacTissueElements(coords, elems, "Triangle")
tissue.conductivity = conductivity
tissue.fibers = fibers
# tissue.mesh += (np.random.random(coords.shape[0]) < 0.2)

# print(tissue.mesh.shape)

# create model object and set up parameters

stim_indexes = np.random.choice(coords.shape[0], size=10, replace=False)
stim_coords = coords[stim_indexes, :]

stim_point_0 = coords[19600:19601]
stim_point_1 = coords[72902:72903]

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimCurrentElectrodes(0, 15, 0.1, geodesic.points, 1))
stim_sequence.add_stim(fw.StimCurrentElectrodes(22, 15, 0.1, stim_point_1, 1))
# stim_sequence.add_stim(fw.StimCurrentElectrodes(55, 15, 0.1, geodesic.points, 1))
# stim_sequence.add_stim(fw.StimCurrentElectrodes(100, 15, 0.1, stim_point_1, 1))

# for stim_time in [0, 26, 52, 78]:
#     stim_sequence.add_stim(fw.StimCurrentElectrodes(stim_time, 30, 0.1, coords[33128:33129], 3))
#     stim_sequence.add_stim(fw.StimCurrentElectrodes(stim_time, 30, 0.1, coords[30639:30640], 3))
#     # stim_sequence.add_stim(fw.StimCurrentElectrodes(stim_time, 15, 0.1, coords[13372:13373], 1))

state_point = coords[75398]
dist = np.linalg.norm(coords - state_point, axis=1)
state_indexes = np.where(dist < 8)[0]
state_coords = coords[state_indexes]

lat_tracker = fw.LocalActivationTimeTracker(step=10, start_time=70)
tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(lat_tracker)


prepacing_sequence = [
    {"n_beats": 30,
     "cycle_length": 30,
     "stim_duration": 0.1,
     "stim_amplitude": 20.,
     "dt": 0.01},
]
cardiac_model = fw.AlievPanfilov()
cardiac_model.prepacing(prepacing_sequence)

# create model object and set up parameters:
simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.t_max = 100
# simulation.state_loader = fw.StateLoader(path)
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = cardiac_model
simulation.tracker_sequence = tracker_sequence
simulation.stim_sequence = stim_sequence
# simulation.stencil = stencil

# u = np.load(path / "u.npy")
# v = np.load(path / "v.npy")

# u[state_indexes] = 0.
# v[state_indexes] = 1.5

# run the model:
simulation.run(num_of_threads=6, initialize=True)

u = simulation.cardiac_model.u
v = simulation.cardiac_model.v
np.save(path / "u_x3.npy", u)
np.save(path / "v_x3.npy", v)

# plt.plot(simulation.solver.num_iterations)
# plt.xlabel("Time Step")
# plt.ylabel("Number of Iterations")
# plt.title("Convergence of CG Solver")
# plt.show()

# # # show the potential map at the end of calculations:
# faces = np.hstack([[3, *tri] for tri in elems])
# mesh = pv.PolyData(coords, faces)
# mesh.point_data["values"] = u
# # mesh.plot(cmap="RdBu_r", show_edges=True)

# def callback(point):
#     # Get closest point ID
#     point_id = mesh.find_closest_point(point)
#     print("Picked point ID:", point_id)

lat_map = lat_tracker.activation_map(70, 100)
np.save(path / "lat_map_x3.npy", lat_map)

# pickable plot
plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=False, scalars=lat_map, cmap="magma")
plotter.add_mesh(geodesic, color="red", line_width=3)
plotter.add_points(stim_point_0, color="blue", point_size=10,
                   render_points_as_spheres=True)
plotter.add_points(stim_point_1, color="blue", point_size=10,
                   render_points_as_spheres=True)
# plotter.enable_point_picking(callback=callback, show_point=True)
plotter.show()
