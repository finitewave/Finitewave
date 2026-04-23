import numpy as np
import pyvista as pv

import finitewave as fw


# create a tissue of size 50x50 with 200x200 points:
n = 200
size = 50
coords, elems = fw.build_triangulated_mesh(n, n, (0, size), (0, size))

# create cardiac tissue object:
tissue = fw.CardiacTissueElements(coords, elems, elem_type='Triangle')
tissue.mesh += (np.random.rand(*tissue.mesh.shape) < 0.2).astype(int)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(0, 1, 0, size, 0, 1))
stim_sequence.add_stim(fw.StimVoltageCoord(45, 1, 0, size//2, 0, size))

# create model object and set up parameters:
simulation = fw.CardiacSimulation(backend="numba")
simulation.dt = 0.01
simulation.t_max = 30
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.AlievPanfilov()
simulation.stim_sequence = stim_sequence
# set up the solver:
# simulation.solver = fw.ForwardEulerSolver()

# run the model:
simulation.run()

# get the resulting potential at the element centers:
u = simulation.cardiac_model.output("u")
elems_u = np.zeros(tissue.elems.shape[0]) * np.nan
elems_u[tissue.myo_elems_indexes] = u[tissue.myo_elements].mean(axis=1)
coords_3d = np.hstack([coords, np.zeros((coords.shape[0], 1))])

# show the potential map at the end of calculations:
faces = np.hstack([[elems.shape[1], *elem] for elem in elems])
mesh = pv.PolyData(coords_3d, faces)
# mesh.cell_data["u"] = elems_u
mesh["u"] = u

pl = pv.Plotter()
pl.add_mesh(mesh, cmap="RdBu_r", show_edges=False, nan_color="white")
pl.camera_position = 'xy'
pl.show()
