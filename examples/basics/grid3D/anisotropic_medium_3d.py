"""
3D Cube with Anisotropic Medium
================================

This example demonstrates how to anisotropic affect on the wave propagation 
in a 3D cube.

A central stimulus initiates activation, and the resulting wave propagation

Fiber Setup:
------------
- Domain size: 200×200×200 (i, j, k)
- Fiber rotation:
    • phi = -pi/4
    • theta = pi/3

Model & Stimulation:
--------------------
- Model: Aliev-Panfilov 3D
- Time: 15 time units total
- Stimulus:
    • Applied at the center of the cube
    • Time: t = 0
    • Strength: 1 (voltage)

Numerical Setup:
----------------
- Time step (dt): 0.01
- Space step (dr): 0.25

Visualization:
--------------
- The cube is rendered using `VisMeshBuilder3D` with opacity 0.3
- The front of the wave (``u > 0.95``) is shown.

Applications:
-------------
- Studying anisotropic conduction in 3D
"""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import finitewave as fw


# number of nodes on the side
n_i = 100
n_j = 100
n_k = 100

phi = - np.pi / 4
theta = np.pi / 3

# set up the cardiac tissue:
tissue = fw.CardiacTissueGrid((n_i, n_j, n_k))

# add fibers orientation vectors with rotation angle phi and theta
tissue.fibers = np.zeros((n_i, n_j, n_k, 3))
tissue.fibers[:, :, :, 0] = np.cos(phi) * np.cos(theta)
tissue.fibers[:, :, :, 1] = np.sin(phi) * np.cos(theta)
tissue.fibers[:, :, :, 2] = np.sin(theta)

diffusion_model = fw.GridModel()
diffusion_model.stencil = fw.CellStencil()

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(0, 1,
                                           n_i // 2 - 3, n_i // 2 + 3,
                                           n_j // 2 - 3, n_j // 2 + 3,
                                           n_k // 2 - 3, n_k // 2 + 3))
# create model object:
simulation = fw.CardiacSimulation()
# set up numerical parameters:
simulation.dt = 0.01
simulation.dr = 0.25
simulation.t_max = 30
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.AlievPanfilov()
simulation.stim_sequence = stim_sequence
simulation.diffusion_model = diffusion_model
# initialize model: compute weights, add stimuls, trackers etc.
simulation.run()

# visualize the potential map in 3D at the end of calculations:
u = simulation.cardiac_model.u

mesh_builder = fw.VisMeshBuilder3D()
grid = mesh_builder.build_mesh(u > 0.95)
grid = mesh_builder.add_scalar(u, 'u')

full_mesh_builder = fw.VisMeshBuilder3D()
full_grid = full_mesh_builder.build_mesh(tissue.mesh)
full_grid = full_mesh_builder.add_scalar(u, 'u')

pl = pv.Plotter()
pl.add_mesh(grid, scalars='u', cmap='RdBu_r')
pl.add_mesh(full_grid, scalars='u', cmap='RdBu_r', opacity=0.3)
pl.show()
