"""
Running the Luo-Rudy 1991 Model in 3D Cardiac Tissue
====================================================

Overview:
---------
This example demonstrates how to run a 3D simulation of the 
Luo-Rudy 1991 ventricular action potential model using the Finitewave framework.
It simulates wave propagation in cardiac tissue in response to a stimulus.

Simulation Setup:
-----------------
- Tissue Grid: A 100×100×10 cardiac tissue domain.
- Stimulation:
  - A planar stimulus is applied along the top edge of the domain at t = 0 ms
    to initiate wavefront propagation.
- Time and Space Resolution:
  - Temporal step (dt): 0.01 ms
  - Spatial resolution (dr): 0.25 mm
  - Total simulation time (t_max): 50 ms

Execution:
----------
1. Create a 3D cardiac tissue grid.
2. Apply a stimulus along the upper boundary to initiate excitation.
3. Set up and run the Luo-Rudy 1991 model.
4. Visualize the transmembrane potential (`u`) at the final time step.

Application:
------------
- Demonstrates how to use more detailed biophysical models.

Output:
-------
A color map of the membrane potential distribution (`u`) is displayed at 
the end of the simulation using matplotlib.

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
luo_rudy = fw.LuoRudy913D()
luo_rudy.dt = 0.01
luo_rudy.dr = 0.25
luo_rudy.t_max = 30

# add the tissue and the stim parameters to the model object
luo_rudy.cardiac_tissue = tissue
luo_rudy.stim_sequence = stim_sequence

# run the model
luo_rudy.run()

# Visualization of selected slices (z = 2, 5, 8)
slice_z_indices = [2, 5, 8]
fig, axs = plt.subplots(1, len(slice_z_indices), figsize=(15, 5))

for i, z in enumerate(slice_z_indices):
    ax = axs[i]
    u = luo_rudy.u[:, :, z]
    im = ax.imshow(u, cmap='viridis', origin='lower',
                   vmin=np.min(u), vmax=np.max(u))
    ax.set_title(f"Slice at z = {z}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

# Add colorbar
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7)
cbar.set_label("Membrane potential (u)")

plt.show()