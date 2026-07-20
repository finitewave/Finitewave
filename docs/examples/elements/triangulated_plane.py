import numpy as np
import pyvista as pv

import finitewave as fw


# create a tissue of size 50x50 with 200x200 points:
n = 200
size = 50
elem_type = fw.ElementType.TRIANGLE
coords, elems = fw.build_triangulated_plane(n, n, (0, size), (0, size))

# create cardiac tissue object:
tissue = fw.CardiacTissueElements(coords, elems, elem_type=elem_type)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(0, 1, 0, size, 0, 1))
stim_sequence.add_stim(fw.StimVoltageCoord(45, 1, 0, size//2, 0, size))

# create model object and set up parameters:
simulation = fw.CardiacSimulation(backend="jax")
simulation.dt = 0.01
simulation.t_max = 100
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.AlievPanfilov()
simulation.stim_sequence = stim_sequence
# set up the solver:
simulation.solver = fw.ForwardEulerSolver()

# run the model:
simulation.run()

# get the resulting potential at the element centers:
u = simulation.cardiac_model.output("u")

# show the potential map at the end of calculations:
mesh = fw.PyVistaSurfaceGrid(coords, elems)
mesh["u"] = u

pl = pv.Plotter()
pl.add_mesh(mesh, cmap="RdBu_r", show_edges=False, nan_color="white")
pl.camera_position = 'xy'
pl.show()
