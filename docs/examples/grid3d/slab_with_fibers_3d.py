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
- Domain size: 200×200×50 (i, j, k)
- Fiber rotation:
    • Varies linearly from -π/3 to +π/3 along the k-axis (depth)
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
- The slab is rendered using `PyVistaMeshGrid` with `as_surface=True` to show the outer surface.
- The upper half is clipped away for a better internal view
- Voltage (`u`) is shown using a colormap

Applications:
-------------
- Mimics realistic ventricular transmural fiber rotation
- Useful for studying anisotropic conduction, twist in scroll waves,
  and depth-dependent activation patterns
"""


import finitewave as fw

import numpy as np
import pyvista as pv


# number of nodes on the side
n_i = 128
n_j = 128
n_k = 50

# set up the cardiac tissue:
tissue = fw.CardiacTissue((n_i, n_j, n_k), dr=0.25)
# orientation of fibers changes along the z-axis from -pi/3 to pi/3
phi_k = np.linspace(- np.pi / 3, np.pi / 3, n_k)
# add fibers orientation vectors
tissue.fibers = np.zeros((n_i, n_j, n_k, 3))
for k, phi in enumerate(phi_k):
    tissue.fibers[:, :, k, 0] = np.cos(phi)
    tissue.fibers[:, :, k, 1] = np.sin(phi)
    tissue.fibers[:, :, k, 2] = 0

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(0, 1,
                                           n_i // 2 - 3, n_i // 2 + 3,
                                           n_j // 2 - 3, n_j // 2 + 3,
                                           #    n_k // 2 - 3, n_k // 2 + 3))
                                           0, n_k))

# create model object:
simulation = fw.CardiacSimulation()
# set up numerical parameters:
simulation.dt = 0.01
simulation.t_max = 10
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.AlievPanfilov()
simulation.stim_sequence = stim_sequence
# initialize and run simulation: compute weights, add stimuls, trackers etc.
simulation.run()

u = simulation.cardiac_model.output("u")
mesh = tissue.mesh
# visualize the wavefront at the end of calculations:

full_grid = fw.PyVistaMeshGrid(mesh, as_surface=True)
full_grid["u"] = u
grid = fw.PyVistaMeshGrid(u > 0.1, as_surface=True)
grid["u"] = simulation.cardiac_model.u

# set transparent background for better visualization of the internal structure
pv.global_theme.transparent_background = True
pl = pv.Plotter()
pl.add_mesh(grid, scalars='u', cmap='RdBu_r', clim=[0, 1])
pl.add_mesh(full_grid, scalars='u', cmap='RdBu_r', clim=[0, 1], opacity=0.3)
pl.show()

pl.screenshot("slab_with_fibers.png")
