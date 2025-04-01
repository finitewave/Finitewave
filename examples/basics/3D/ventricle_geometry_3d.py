"""
LV Simulation
-------------

This example demonstrates usage of the left ventricle mesh and fibers from
data storage (https://zenodo.org/records/3890034) and the Aliev-Panfilov model.
"""

from pathlib import Path
import numpy as np

import finitewave as fw


path = Path(__file__).parent

# Load mesh as cubic array
mesh = np.load(path.joinpath("data", "mesh.npy"))

# Load fibers as cubic array
fibers_list = np.load(path.joinpath("data", "fibers.npy"))
fibers = np.zeros(mesh.shape + (3,), dtype=float)
fibers[mesh > 0] = fibers_list

# set up the tissue with fibers orientation:
tissue = fw.CardiacTissue3D(mesh.shape)
tissue.mesh = mesh
tissue.add_boundaries()
tissue.fibers = fibers

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, mesh.shape[0],
                                             0, mesh.shape[0],
                                             0, 20))

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 40
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
# initialize model: compute weights, add stimuls, trackers etc.
aliev_panfilov.run()

# visualize the ventricle in 3D
mesh_builder = fw.VisMeshBuilder3D()
mesh_grid = mesh_builder.build_mesh(tissue.mesh)
mesh_grid = mesh_builder.add_scalar(aliev_panfilov.u, 'u')
mesh_grid.plot(clim=[0, 1], cmap='viridis')