import numpy as np
import pyvista as pv

import finitewave as fw


# create a tissue of size 400x400 with cardiomycytes:
nx = 200
ny = 200
nz = 10
size_x = (0, 50)
size_y = (0, 50)
size_z = (0, 2.5)

coords, elems = fw.build_tetrahedral_mesh(nx, ny, nz, size_x, size_y, size_z)

# create cardiac tissue object:
tissue = fw.CardiacTissueElements(coords, elems, fw.ElementType.TETRA)
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(0, 1,
                                           0, size_x[1],
                                           0, 1,
                                           0, size_z[1]))

# create model object and set up parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=50, backend="jax")
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.Courtemanche()
simulation.stim_sequence = stim_sequence
simulation.solver = fw.ForwardEulerSolver()

# run the model:
simulation.run()

# show the potential map at the end of calculations:
grid = fw.PyVistaTetraGrid(coords, elems, as_surface=True)
grid["u"] = simulation.cardiac_model.u

plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="u", cmap="RdBu_r")
plotter.show()
