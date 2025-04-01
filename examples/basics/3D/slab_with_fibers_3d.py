"""
Slab with rotating fibers
==============================

This example demonstrates how to create a 3D slab with rotating fibers.
"""

import finitewave as fw

import matplotlib.pyplot as plt
import numpy as np


# number of nodes on the side
n_i = 200
n_j = 200
n_k = 100

# set up the cardiac tissue:
tissue = fw.CardiacTissue3D((n_i, n_j, n_k))
# orientation of fibers changes along the z-axis from -pi/3 to pi/2
phi_k = np.linspace(- np.pi / 3, np.pi / 2, n_k - 2)
# add fibers orientation vectors
tissue.fibers = np.zeros((n_i, n_j, n_k, 3))
for k, phi in enumerate(phi_k):
    tissue.fibers[:, :, k + 1, 0] = np.cos(phi)
    tissue.fibers[:, :, k + 1, 1] = np.sin(phi)
    tissue.fibers[:, :, k + 1, 2] = 0

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1,
                                             n_i // 2 - 5, n_i // 2 + 5,
                                             n_j // 2 - 5, n_j // 2 + 5,
                                             0, n_k))
# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 15
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
# initialize model: compute weights, add stimuls, trackers etc.
aliev_panfilov.run()

# visualize the potential map in 3D
vis_mesh = tissue.mesh.copy()
vis_mesh[n_i//2:, n_j//2:, :] = 0

mesh_builder = fw.VisMeshBuilder3D()
grid = mesh_builder.build_mesh(vis_mesh)
grid = mesh_builder.add_scalar(aliev_panfilov.u, 'u')
grid.plot(clim=[0, 1], cmap='viridis')