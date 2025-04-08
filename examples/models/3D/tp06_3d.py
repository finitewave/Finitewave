"""
Running the TP06 Model in 3D Cardiac Tissue
===========================================

Overview:
---------
This example demonstrates how to run a 3D simulation of the 
ten Tusscher–Panfilov 2006 (TP06) model for ventricular cardiomyocytes 
using the Finitewave framework. The TP06 model provides a detailed 
biophysical representation of cardiac electrophysiology.

Simulation Setup:
-----------------
- Tissue Grid: A 300×300×10 cardiac tissue domain.
- Stimulation:
  - A planar stimulus is applied along the top edge (rows 0 to 5) at t = 0 ms
    to initiate wave propagation.
- Time and Space Resolution:
  - Temporal step (dt): 0.01 ms
  - Spatial resolution (dr): 0.25 mm
  - Total simulation time (t_max): 50 ms

Execution:
----------
1. Create a 3D cardiac tissue grid.
2. Apply a stimulus to initiate excitation.
3. Set up and run the TP06 model.
4. Visualize the final membrane potential (`u`) distribution.

Application:
------------
- Demonstrates how to use more detailed biophysical models.

Output:
-------
Displays the membrane potential at the final time step using matplotlib.

"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import finitewave as fw

n = 100
nk = 10
# create mesh
tissue = fw.CardiacTissue3D((n, n, nk))

# set up stimulation parameters
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, n, 0, 5, 0, nk))

# create model object and set up parameters
tp06 = fw.TP063D()
tp06.dt = 0.01
tp06.dr = 0.25
tp06.t_max = 30

# add the tissue and the stim parameters to the model object
tp06.cardiac_tissue = tissue
tp06.stim_sequence = stim_sequence

# run the model
tp06.run()

# Visualization of selected slices (z = 2, 5, 8)
slice_z_indices = [2, 5, 8]
fig, axs = plt.subplots(1, len(slice_z_indices), figsize=(15, 5))

for i, z in enumerate(slice_z_indices):
    ax = axs[i]
    u = tp06.u[:, :, z]
    im = ax.imshow(u, cmap='viridis', origin='lower',
                   vmin=np.min(u), vmax=np.max(u))
    ax.set_title(f"Slice at z = {z}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

# Add colorbar
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7)
cbar.set_label("Membrane potential (u)")

plt.show()