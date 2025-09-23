"""
Spiral Waves on a 3D Spherical Shell
====================================

This example demonstrates how to simulate spiral (scroll) waves inside
a 3D spherical shell using the Aliev-Panfilov model with Finitewave.

A hollow sphere is embedded inside a 3D Cartesian grid. The propagation
of electrical activity is initiated by sequential stimuli, creating a
scroll wave that circulates within the curved geometry.

The resulting potential distribution is visualized with Finitewave's
3D mesh tools.

Geometry Setup:
---------------
- Domain size: 200×200×200 grid
- Geometry: Spherical shell created using a binary mask
    - Outer radius: 95 voxels
    - Inner radius: 90 voxels
    - Mesh values: 1 inside the shell, 0 outside
- The sphere is centered in the domain

Stimulation Protocol:
---------------------
- Stimulus 1:
    - Time: t = 0
    - Location: One side of the sphere (thin planar region near the edge)
- Stimulus 2:
    - Time: t = 50
    - Location: One hemisphere only
- This breaks the initial wave symmetry and initiates a scroll wave

Model:
------
- Aliev-Panfilov 3D reaction-diffusion model
- Time step (dt): 0.01
- Space step (dr): 0.25
- Total simulation time: 100

Visualization:
--------------
The 3D scalar field (`u`) is rendered on the shell mesh using
Finitewave’s `VisMeshBuilder3D`.

Applications:
-------------
- Simulation of scroll wave dynamics in spherical domains
- Study of wave breakups, phase singularities, and 3D reentry
- Modeling electrical activity in simplified anatomical geometries
"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw


# Create a spherical mask within a 100x100x100 cube
def create_sphere_mask(shape, radius, center):
    z, y, x = np.indices(shape)
    distance = np.sqrt((x - center[0])**2 +
                       (y - center[1])**2 +
                       (z - center[2])**2)
    mask = distance <= radius
    return mask


def create_sphere(shape, radius, center):
    mesh = np.zeros(shape)
    mesh[create_sphere_mask(mesh.shape, radius, center)] = 1
    mesh[create_sphere_mask(mesh.shape, radius-5, center)] = 0
    mesh = mesh[:shape[0] - n//4, :, :]
    return mesh


# set up the cardiac tissue:
n = 200
shape = (n, n, n)
mesh = create_sphere(shape, n//2-5, (n//2, n//2, n//2))
n, m, k = mesh.shape

tissue = fw.CardiacTissueFDM((n, m, k))
tissue.mesh = mesh

# set up stimulation parameters:
min_z = np.where(tissue.mesh)[2].min()

stim1 = fw.StimVoltageCoordFDM(0, 1,
                               0, n,
                               0, m,
                               min_z, min_z + 3)

stim2 = fw.StimVoltageCoordFDM(50, 1,
                               0, n,
                               0, m//2,
                               0, k)

stim_sequence = fw.StimSequence()
stim_sequence.add_stim(stim1)
stim_sequence.add_stim(stim2)

aliev_panfilov = fw.AlievPanfilovFDM()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 10
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# visualize the potential map in 3D
vis_mesh = tissue.mesh.copy()
# vis_mesh[n//2:, n//2:, n//2:] = 0

mesh_builder = fw.VisMeshBuilder3D()
grid = mesh_builder.build_mesh(vis_mesh, as_surface=True)
grid = mesh_builder.add_scalar(aliev_panfilov.u, 'u')
grid.plot(clim=[0, 1], cmap='RdBu_r')
