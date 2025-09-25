
#
# Left ventricle simlation with the Aliev-Panfilov model.
# Mesh and fibers were taken from Niderer's data storage (https://zenodo.org/records/3890034)
# Fibers were generated with Rule-based algorithm.
# Ventricle is stimulated from the apex.

from pathlib import Path
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

import finitewave as fw


path = Path(__file__).parent

# Load mesh as cubic array
mesh = np.load(path.joinpath("data", "mesh.npy"))

tissue = fw.CardiacTissue3D(mesh.shape)
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = mesh
# generate 20% of fibrosis in the ventrcile wall:
fibrosis_pattern = fw.Diffuse3DPattern(0, mesh.shape[0], 0, mesh.shape[1], 0, mesh.shape[2], 0.20)
fibrosis_pattern.generate(tissue.mesh.shape, tissue.mesh)

tissue.add_boundaries()

# create model object:
tp06 = fw.TP063D()
# set up numerical parameters:
tp06.dt = 0.01
tp06.dr = 0.25
tp06.t_max = 25
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, -20, 0, mesh.shape[0],
                                             0, mesh.shape[0],
                                             0, 30))
# add the tissue and the stim parameters to the model object:
tp06.cardiac_tissue = tissue
tp06.stim_sequence = stim_sequence
# initialize model: compute weights, add stimuls, trackers etc.
tp06.run()

# show the potential map at the end of calculations

# visualize the ventricle in 3D
mesh_builder = fw.VisMeshBuilder3D()
mesh_grid = mesh_builder.build_mesh(tissue.mesh)
mesh_grid = mesh_builder.add_scalar(tp06.u, 'u')
mesh_grid.plot(clim=[-80, 30], cmap='viridis')
