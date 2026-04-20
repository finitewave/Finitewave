
#
# Left ventricle simlation with the Aliev-Panfilov model.
# Mesh and fibers were taken from Niderer's data storage (https://zenodo.org/records/3890034)
# Fibers were generated with Rule-based algorithm.
# Ventricle is stimulated from the apex.

from pathlib import Path
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

import finitewave.mlxwave as fw


path = Path(__file__).parent.parent

# Load mesh as cubic array
mesh = np.load(path.joinpath("data", "mesh.npy"))

tissue = fw.CardiacTissueGrid(mesh.shape, dr=0.25)
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = mesh
# # generate 20% of fibrosis in the ventrcile wall:
# fibrosis_pattern = fw.Diffuse3DPattern(0, mesh.shape[0], 0, mesh.shape[1], 0, mesh.shape[2], 0.20)
# fibrosis_pattern.generate(tissue.mesh.shape, tissue.mesh)

# create model object:
tp06 = fw.TenTusscherPanfilov2006()
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(0, -20,
                                           0, mesh.shape[0],
                                           0, mesh.shape[0],
                                           0, 30))
# add the tissue and the stim parameters to the model object:
simulation = fw.CardiacSimulation(dt=0.01, t_max=0.01)
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence
# initialize model: compute weights, add stimuls, trackers etc.
simulation.cardiac_model = tp06
simulation.run()

# show the potential map at the end of calculations

# # visualize the ventricle in 3D
# grid = fw.PyVistaMeshGrid(tissue.mesh, as_surface=True)
# grid["u"] = simulation.cardiac_model.u
# grid.plot(scalars="u", cmap="inferno", show_edges=False, show_scalar_bar=False)
