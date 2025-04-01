"""
Sphere with spiral waves in 3D
==============================

This example demonstrates how to create a 3D sphere with spiral waves.
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


# set up the cardiac tissue:
n = 200
shape = (n, n, n)
tissue = fw.CardiacTissue3D((n, n, n))
tissue.mesh = np.zeros((n, n, n))
tissue.mesh[create_sphere_mask(tissue.mesh.shape,
                               n//2-5,
                               (n//2, n//2, n//2))] = 1
tissue.mesh[create_sphere_mask(tissue.mesh.shape,
                               n//2-10,
                               (n//2, n//2, n//2))] = 0

# set up stimulation parameters:
min_x = np.where(tissue.mesh)[0].min()

stim1 = fw.StimVoltageCoord3D(0, 1,
                              min_x, min_x + 3,
                              0, n,
                              0, n)

stim2 = fw.StimVoltageCoord3D(50, 1,
                              0, n,
                              0, n//2,
                              0, n)

stim_sequence = fw.StimSequence()
stim_sequence.add_stim(stim1)
stim_sequence.add_stim(stim2)

aliev_panfilov = fw.AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 100
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# visualize the potential map in 3D
vis_mesh = tissue.mesh.copy()
# vis_mesh[n//2:, n//2:, n//2:] = 0

mesh_builder = fw.VisMeshBuilder3D()
grid = mesh_builder.build_mesh(vis_mesh)
grid = mesh_builder.add_scalar(aliev_panfilov.u, 'u')
grid.plot(clim=[0, 1], cmap='viridis')