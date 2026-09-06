import numpy as np
import pyvista as pv

import finitewave as fw


# create a tissue of size 400x400 with cardiomycytes:
nx, ny, nz = 200, 100, 20
size_x = (0, 50)
size_y = (0, 25)
size_z = (0, 5)

coords, elems = fw.build_tetrahedral_slab(nx, ny, nz, size_x, size_y, size_z)

# create cardiac tissue object:
tissue = fw.CardiacTissueElements(coords, elems, fw.ElementType.TETRA)
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(0, 1,
                                           0, 1,
                                           0, size_y[1],
                                           0, size_z[1]))

# create model object and set up parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=10, backend="mlx")
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.FentonKarma()
simulation.stim_sequence = stim_sequence
# Set up time integration. BackwardEulerTimeIntegration is used by default:
simulation.time_integration = fw.BackwardEulerTimeIntegration(atol=1e-6, maxiter=100)

# run the model:
simulation.run()

# import matplotlib.pyplot as plt

# plt.plot(simulation.time_integration.num_iterations)
# plt.show()

# show the potential map at the end of calculations:
grid = fw.PyVistaTetraGrid(coords, elems, as_surface=True)
grid["u"] = simulation.cardiac_model.output("u")

plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="u", cmap="RdBu_r")
plotter.show()

# plotter.screenshot("tetrahedral_slab.png")
