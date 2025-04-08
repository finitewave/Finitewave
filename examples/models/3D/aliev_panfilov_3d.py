"""
Running the Aliev-Panfilov Model in 3D
======================================

Overview:
---------
This example demonstrates how to run a basic 3D simulation of the 
Aliev-Panfilov model using the Finitewave framework. It simulates 
electrical excitation in cardiac tissue following a localized stimulus.

Simulation Setup:
-----------------
- Tissue Grid: A 300×300×10 cardiac tissue domain.
- Stimulation:
  - A square side stimulus is applied at t = 0.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 20

Execution:
----------
1. Create a 3D cardiac tissue grid.
2. Apply a stimulus along the upper boundary to initiate excitation.
3. Set up and run the Aliev-Panfilov model.
4. Visualize the final transmembrane potential map.

Application:
------------
- This example serves as a minimal working simulation for testing and demonstration.
- It can be used as a template for:
  - Adding trackers
  - Extending to anisotropic or heterogeneous tissues
  - Exploring basic wave propagation

Output:
-------
Displays the membrane potential distribution (`u`) at the end of the simulation using matplotlib.

"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import finitewave as fw
from mpl_toolkits.mplot3d import Axes3D

# create a tissue of size 300x300 with cardiomycytes:
n = 100
nk = 10
tissue = fw.CardiacTissue3D([n, n, nk])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, n, 0, 5, 0, nk))

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilov3D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 5
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

# run the model:
aliev_panfilov.run()

# Visualization of selected slices (z = 2, 5, 8)
slice_z_indices = [2, 5, 8]
fig, axs = plt.subplots(1, len(slice_z_indices), figsize=(15, 5))

for i, z in enumerate(slice_z_indices):
    ax = axs[i]
    u = aliev_panfilov.u[:, :, z]
    im = ax.imshow(u, cmap='viridis', origin='lower',
                   vmin=np.min(u), vmax=np.max(u))
    ax.set_title(f"Slice at z = {z}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

# Add colorbar
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7)
cbar.set_label("Membrane potential (u)")

plt.show()