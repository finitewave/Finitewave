
#
# The model is a 3D Aliev-Panfilov model with anisotropic stencil.
# The model is stimulated with a voltage pulse in the center of the tissue.
# Anisotropy is set by specifying a fiber array (CardiacTissue class) and
# diffusion coefficients D_al, D_ac (diffusion along and across fibers).
#

"""
AlievPanfilov3D (Aniso)
==========================

This example demonstrates how to use the Aliev-Panfilov model in 3D with
anisotropic stencil.

.. note::

    In anisotropic models, the ``AssymmetricStencil3D`` class should be used.
"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# set up the tissue with fibers orientation:
n = 100
tissue = fw.CardiacTissue3D((n, n, n))
# add fibers orientation vectors
theta, alpha = 0. * np.pi, 0.25 * np.pi
tissue.fibers = np.zeros((n, n, n, 3))
tissue.fibers[..., 0] = np.cos(theta) * np.cos(alpha)
tissue.fibers[..., 1] = np.cos(theta) * np.sin(alpha)
tissue.fibers[..., 2] = np.sin(theta)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1,
                                             n//2 - 5, n//2 + 5,
                                             n//2 - 5, n//2 + 5,
                                             n//2 - 5, n//2 + 5))

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 10
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
# initialize model: compute weights, add stimuls, trackers etc.
aliev_panfilov.run()

vis_mesh = tissue.mesh.copy()
vis_mesh[n//2:, n//2:, n//2:] = 0

mesh_builder = fw.VisMeshBuilder3D()
grid = mesh_builder.build_mesh(vis_mesh)
grid = mesh_builder.add_scalar(aliev_panfilov.u, 'u')
grid.plot(clim=[0, 1], cmap='viridis')
