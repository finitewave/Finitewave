
#
# Left ventricle simlation with the Aliev-Panfilov model.
# Mesh and fibers were taken from Niderer's data storage (https://zenodo.org/records/3890034)
# Here we use matrix stimlation to simultaneusly stimulate ventricle from apex and base.
# After the end of the simulation you will see two waves propagating from the apex and the base.

from pathlib import Path
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

import finitewave as fw


# path = Path(__file__).parent
path = Path("/home/arstan/Projects/Fibrosis/Finitewave/_examples/")

# Load mesh as cubic array
mesh = np.load(path.joinpath("data", "mesh.npy"))
mesh[-1, :, :] = 0

coords = np.argwhere(mesh > 0)
apex_coords = coords[coords[:, 0] == coords[:, 0].max()]

mesh_builder = fw.VisMeshBuilder3D()
mesh_grid = mesh_builder.build_mesh(mesh)
mesh_grid = mesh_builder.add_scalar(mesh, 'u')
mesh_grid.plot(clim=[0, 3], cmap='viridis')

tissue = fw.CardiacTissue3D(mesh.shape)
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = mesh

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.0015
aliev_panfilov.dr = 0.1
aliev_panfilov.t_max = 15

# set up stimulation parameters:
stim_sequence = fw.StimSequence()

stim_sequence.add_stim(fw.StimCurrentArea3D(0, 10, 1., apex_coords, u_max=1.))

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
# initialize model: compute weights, add stimuls, trackers etc.
aliev_panfilov.run()

# show the potential map at the end of calculations

# visualize the ventricle in 3D
mesh_builder = fw.VisMeshBuilder3D()
mesh_grid = mesh_builder.build_mesh(tissue.mesh)
mesh_grid = mesh_builder.add_scalar(aliev_panfilov.u, 'u')
mesh_grid.plot(clim=[0, 1], cmap='viridis')
