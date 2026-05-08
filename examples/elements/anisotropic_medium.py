import numpy as np
import pyvista as pv

import finitewave as fw


# create a tissue of size 50x50 with 400x400 nodes:
n = 400   # number of nodes (myocytes) along x and y axes
size = 50 # mm
elem_type = fw.ElementType.TRIANGLE

coords, elems = fw.build_triangulated_plane(n, n, (0, size), (0, size))

# create cardiac tissue object:
tissue = fw.CardiacTissueElements(coords, elems, elem_type=elem_type)
# define fiber directions for each ELEMENT.
tissue.fibers = np.zeros((elems.shape[0], 2))
tissue.fibers[:, 0] = np.cos(np.pi / 6) 
tissue.fibers[:, 1] = np.sin(np.pi / 6)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(0, 1,
                                           size//2 - 1, size//2 + 1,
                                           size//2 - 1, size//2 + 1))

stim_sequence.add_stim(fw.StimVoltageCoord(50, 1,
                                           size//2 - 1, size//2 + 1,
                                           size//2 - 1, size//2 + 1))

# create model object and set up parameters:
simulation = fw.CardiacSimulation(backend="jax")
simulation.dt = 0.01
simulation.t_max = 33
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.BuenoOrovio()
simulation.stim_sequence = stim_sequence
# run the model:
simulation.run()

# get the resulting potential at the element centers:
u = simulation.cardiac_model.u

mesh = fw.PyVistaSurfaceGrid(coords, elems)
mesh["u"] = u

pl = pv.Plotter()
pl.add_mesh(mesh, cmap="magma", show_edges=False, nan_color="lightgray")
pl.camera_position = 'xy'
pl.show()
