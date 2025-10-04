"""
3D Slab with Rotating Fibers
============================

This example demonstrates how to create a 3D slab of cardiac tissue
with smoothly rotating fiber orientation along the depth (z-axis).
Such setups are used to mimic myocardial fiber architecture in 
ventricular walls, where fiber orientation rotates across the wall.

A central stimulus initiates activation, and the resulting 
wave propagation is influenced by the local fiber direction at each depth.

Fiber Setup:
------------
- Domain size: 200×200×100 (i, j, k)
- Fiber rotation:
    • Varies linearly from -π/3 to +π/2 along the k-axis (depth)
    • In-plane rotation only (z-component of fibers = 0)
    • Represented as 3D unit vectors: (cos(ϕ), sin(ϕ), 0)

Model & Stimulation:
--------------------
- Model: Mitchell-Schaeffer 3D
- Time: 15 time units total
- Stimulus:
    • Applied at the center of the i-j plane
    • Extends fully along the z-axis (column stimulation)
    • Time: t = 0
    • Strength: 1 (voltage)

Numerical Setup:
----------------
- Time step (dt): 0.01
- Space step (dr): 0.25

Visualization:
--------------
- The slab is rendered using `VisMeshBuilder3D`
- The upper half is clipped away for a better internal view
- Voltage (`u`) is shown using a colormap

Applications:
-------------
- Mimics realistic ventricular transmural fiber rotation
- Useful for studying anisotropic conduction, twist in scroll waves,
  and depth-dependent activation patterns
"""


import finitewave.gridywave as fw

import matplotlib.pyplot as plt
import numpy as np


# number of nodes on the side
n_i = 200
n_j = 200
n_k = 100

# set up the cardiac tissue:
tissue = fw.CardiacTissueGrid((n_i, n_j, n_k))
# orientation of fibers changes along the z-axis from -pi/3 to pi/3
phi_k = np.linspace(- np.pi / 3, np.pi / 3, n_k - 2)
# add fibers orientation vectors
tissue.fibers = np.zeros((n_i, n_j, n_k, 3))
for k, phi in enumerate(phi_k):
    tissue.fibers[:, :, k + 1, 0] = np.cos(phi)
    tissue.fibers[:, :, k + 1, 1] = np.sin(phi)
    tissue.fibers[:, :, k + 1, 2] = 0

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageGridCoord(0, 1,
                                               n_i // 2 - 5, n_i // 2 + 5,
                                               n_j // 2 - 5, n_j // 2 + 5))
# create model object:
simulation = fw.CardiacGridSimulation()
# set up numerical parameters:
simulation.dt = 0.01
simulation.dr = 0.25
simulation.t_max = 15
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.AlievPanfilov()
simulation.stim_sequence = stim_sequence
# initialize model: compute weights, add stimuls, trackers etc.
simulation.run()

# visualize the potential map in 3D
vis_mesh = tissue.mesh.copy()
vis_mesh[n_i//2:, n_j//2:, :] = 0

mesh_builder = fw.VisMeshBuilder3D()
grid = mesh_builder.build_mesh(vis_mesh)
grid = mesh_builder.add_scalar(simulation.cardiac_model.u, 'u')
grid.plot(cmap='RdBu_r')
