import numpy as np
import pyvista as pv

import finitewave as fw


# create a tissue of size 50x50 with 200x200 points:
n = 400
size = 50
coords, elems = fw.build_triangulated_mesh(n, n, (0, size), (0, size))

# create cardiac tissue object:
tissue = fw.CardiacTissueElements(coords, elems, elem_type=fw.ElementType.TRIANGLE)
# tissue.mesh += (np.random.rand(*tissue.mesh.shape) < 0.1).astype(int)
# define fiber directions for anisotropic conduction:
tissue.fibers = np.zeros((elems.shape[0], 2))
tissue.fibers[:, 0] = np.cos(np.pi / 6)  # set fiber direction at 30 degrees
tissue.fibers[:, 1] = np.sin(np.pi / 6)  # set fiber direction at 30 degrees

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(0, 1,
                                           size//2 - 1, size//2 + 1,
                                           size//2 - 1, size//2 + 1))

stim_sequence.add_stim(fw.StimVoltageCoord(50, 1,
                                           size//2 - 1, size//2 + 1,
                                           size//2 - 1, size//2 + 1))

# create model object and set up parameters:
simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.t_max = 50
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.BuenoOrovio()
simulation.stim_sequence = stim_sequence

# run the model:
simulation.run()

# get the resulting potential at the element centers:
u = simulation.cardiac_model.u
elems_u = np.mean(u[elems], axis=1)
elems_u[~tissue.myo_elems_mask] = np.nan
coords_3d = np.hstack([coords, np.zeros((coords.shape[0], 1))])

# show the potential map at the end of calculations:
faces = np.hstack([[elems.shape[1], *elem] for elem in elems])
mesh = pv.PolyData(coords_3d, faces)
mesh.cell_data["values"] = elems_u
# mesh.point_data["values"] = u

# import matplotlib.pyplot as plt

# plt.plot(simulation.solver.num_iterations)
# plt.show()

pl = pv.Plotter()
pl.add_mesh(mesh, cmap="magma", show_edges=False, nan_color="lightgray")
pl.camera_position = 'xy'
pl.show()
