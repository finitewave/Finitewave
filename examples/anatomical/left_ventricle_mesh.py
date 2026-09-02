
from pathlib import Path

import numpy as np
import pyvista as pv

import finitewave as fw


# create a tissue of size 400x400 with cardiomycytes:
path = Path(__file__).parent.parent
coords = np.load(path / "data" / "lv_mesh" / "points.npy")
elems = np.load(path / "data" / "lv_mesh" / "cells.npy")

# create cardiac tissue object:
tissue = fw.CardiacTissueElements(coords, elems, fw.ElementType.TETRA)
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimCurrentElectrodes(0, 10, 0.5, coords[:1], size=5))

# create model object and set up parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=1, backend="mlx")
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.FentonKarma()
simulation.stim_sequence = stim_sequence
# set up the solver, if not specified BackwardEulerSolver is used by default:
simulation.solver = fw.BackwardEulerSolver(atol=1e-6)

# run the model:
simulation.run()

import matplotlib.pyplot as plt

plt.plot(simulation.solver.num_iterations)
plt.show()


# # show the potential map at the end of calculations:
# grid = fw.PyVistaTetraGrid(coords, elems, as_surface=True)
# grid["u"] = np.asarray(simulation.cardiac_model.u)

# plotter = pv.Plotter()
# plotter.add_mesh(grid, scalars="u", cmap="RdBu_r")
# plotter.show()

# plotter.screenshot("tetrahedral_slab.png")
