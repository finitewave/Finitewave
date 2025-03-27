"""
Running the TP06 Model in 2D Cardiac Tissue
===========================================

Overview:
---------
This example demonstrates how to run a 2D simulation of the 
ten Tusscher–Panfilov 2006 (TP06) model for ventricular cardiomyocytes 
using the Finitewave framework. The TP06 model provides a detailed 
biophysical representation of cardiac electrophysiology.

Simulation Setup:
-----------------
- Tissue Grid: A 300×300 cardiac tissue domain.
- Stimulation:
  - A planar stimulus is applied along the top edge (rows 0 to 5) at t = 0 ms
    to initiate wave propagation.
- Time and Space Resolution:
  - Temporal step (dt): 0.01 ms
  - Spatial resolution (dr): 0.25 mm
  - Total simulation time (t_max): 50 ms

Execution:
----------
1. Create a 2D cardiac tissue grid.
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
import finitewave as fw

n = 300
# create mesh
tissue = fw.CardiacTissue2D((n, n))

# set up stimulation parameters
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, 5))

# create model object and set up parameters
tp06 = fw.TP062D()
tp06.dt = 0.01
tp06.dr = 0.25
tp06.t_max = 50

# add the tissue and the stim parameters to the model object
tp06.cardiac_tissue = tissue
tp06.stim_sequence = stim_sequence

# run the model
tp06.run()

# show the potential map at the end of calculations:
plt.figure()
plt.imshow(tp06.u)
plt.colorbar()
plt.show()
