from pathlib import Path
import numpy as np
import pyvista as pv
import finitewave as fw
import matplotlib.pyplot as plt


radius = 15
mesh = pv.Icosphere(nsub=7, radius=radius)

coords = mesh.points
elems = mesh.faces.reshape((-1, 4))[:, 1:4]

center1 = np.array([radius, 0, 0])
center2 = np.array([0, radius, 0])

radius_hole = 0.5 * np.pi * radius / 4

elem_centers = np.mean(coords[elems], axis=1)
dist1 = np.linalg.norm(elem_centers - center1, axis=1)
dist2 = np.linalg.norm(elem_centers - center2, axis=1)
mask = (dist1 > radius_hole) & (dist2 > radius_hole)

elems = elems[mask, :]

# phi, theta = np.pi / 4, 0

phi = np.linspace(np.pi/8, 3*np.pi/8, 10)
theta = np.zeros_like(phi)

stim_1_coords = np.array([radius * np.cos(phi) * np.cos(theta),
                          radius * np.sin(phi) * np.cos(theta),
                          radius * np.sin(theta)]).T
# find the closest node to stim_1_coords
idxs = [mesh.find_closest_point(coord) for coord in stim_1_coords]
stim_coords = coords[idxs]
stim_matrix = ((coords[:, 0] > 0) & (coords[:, 1] > 0) & (coords[:, 2] > 0))

# create cardiac tissue object:
tissue = fw.CardiacTissueElements(coords, elems, elem_type='Triangle')

# set up stimulation parameters:
# stim_coords = coords[:1, :]
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageElectrodes(0, 1, stim_1_coords, size=2))
stim_sequence.add_stim(fw.StimCurrentElectrodes(32, 1, 1, stim_1_coords, size=2))
stim_sequence.add_stim(fw.StimVoltageMatrix(54, 1, stim_matrix))

frame_tracker = fw.FrameTracker()
frame_tracker.step = 20
frame_tracker.start_time = 100
tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(frame_tracker)

# create model object and set up parameters:
simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.t_max = 135
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.AlievPanfilov()
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence
# set up the solver:
# simulation.solver = fw.ForwardEulerSolver()

# run the model:
simulation.run()

np.save("coords.npy", coords.astype(np.float32))
np.save("elems.npy", elems.astype(np.int32))

# get the resulting potential at the element centers:
u = simulation.cardiac_model.u
elems_u = np.mean(u[elems], axis=1)

# show the potential map at the end of calculations:
faces = np.hstack([[elems.shape[1], *elem] for elem in elems])
mesh = pv.PolyData(coords, faces)
mesh.cell_data["values"] = elems_u

pl = pv.Plotter()
pl.add_mesh(mesh, cmap="RdBu_r", show_edges=False)
pl.camera_position = 'xy'
pl.show()

# print(f"Number of nodes: {coords.shape[0]}")
# print(f"Number of elements: {elems.shape[0]}")

# pv.set_plot_theme("document")
# p = pv.Plotter()
# p.add_mesh(mesh, color="lightblue", show_edges=True, opacity=0.5)
# p.show()