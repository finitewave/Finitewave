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
- Space step (dr): 0.25 (50x50x50 model units)
- Geometry: Spherical shell created using a binary mask
    - Outer radius: 95 voxels
    - Inner radius: 90 voxels
    - Mesh values: 1 inside the shell, 0 outside
- The sphere is centered in the domain
- The top of the sphere is clipped to mimic an anatomical hole.

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
- Model uses the memory-saving option to reduce memory usage for state variables.

Simulation
----------
- Time step (dt): 0.01
- Total simulation time: 200

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

import numpy as np

import finitewave as fw


def build_sphere_mask(shape, radius, center):
    z, y, x = np.indices(shape)
    distance = np.sqrt((x - center[0])**2 +
                       (y - center[1])**2 +
                       (z - center[2])**2)
    mask = distance <= radius
    return mask


def build_sphere(shape, radius, center):
    mesh = np.zeros(shape)
    mesh[build_sphere_mask(mesh.shape, radius, center)] = 1
    mesh[build_sphere_mask(mesh.shape, radius-5, center)] = 0
    mesh[- n//8:, :, :] = 0
    return mesh


# set up the cardiac tissue:
n = 200
shape = (n, n, n)
mesh = build_sphere(shape, n//2-5, (n//2, n//2, n//2))
n, m, k = mesh.shape

tissue = fw.CardiacTissueGrid((n, m, k), dr=0.25)
tissue.mesh = mesh

# set up stimulation parameters:
z_max = np.where(tissue.mesh)[2].max()
z_min = z_max - 3

stim1 = fw.StimVoltageCoord(time=0, volt_value=1,
                            x_min=0, x_max=n,
                            y_min=0, y_max=m,
                            z_min=z_min, z_max=z_max)
stim2 = fw.StimVoltageCoord(time=50, volt_value=1,
                            x_min=0, x_max=n,
                            y_min=0, y_max=m//2,
                            z_min=0, z_max=k)

stim_sequence = fw.StimSequence()
stim_sequence.add_stim(stim1)
stim_sequence.add_stim(stim2)

stim_prepacing = fw.StimPrepacing(dt=0.01)
stim_prepacing.add_stim(n_beats=5, basic_cycle_length=40, stim_duration=0.1, stim_amplitude=1)
stim_prepacing.add_stim(n_beats=5, basic_cycle_length=30, stim_duration=0.1, stim_amplitude=1)
stim_prepacing.add_stim(n_beats=5, basic_cycle_length=25, stim_duration=0.1, stim_amplitude=1)

model = fw.AlievPanfilov(memory_save=True)
model.prepacing(stim_prepacing)

simulation = fw.CardiacSimulation()
# set up numerical parameters:
simulation.dt = 0.01
simulation.t_max = 200
# add the tissue and the stim parameters to the model object:
simulation.cardiac_model = model
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence

simulation.run()

u = simulation.cardiac_model.u

# visualize the potential map in 3D
grid = fw.PyVistaMeshGrid(tissue.mesh, as_surface=False)
grid['u'] = u
grid.plot(scalars='u', cmap='coolwarm', show_edges=False, show_scalar_bar=True)
