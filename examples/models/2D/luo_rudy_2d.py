"""
Running the Luo-Rudy 1991 Model in 2D Cardiac Tissue
====================================================

Overview:
---------
This example demonstrates how to run a 2D simulation of the 
Luo-Rudy 1991 ventricular action potential model using the Finitewave framework.
It simulates wave propagation in cardiac tissue in response to a stimulus.

Simulation Setup:
-----------------
- Tissue Grid: A 300×300 cardiac tissue domain.
- Stimulation:
  - A planar stimulus is applied along the top edge of the domain at t = 0 ms
    to initiate wavefront propagation.
- Time and Space Resolution:
  - Temporal step (dt): 0.01 ms
  - Spatial resolution (dr): 0.25 mm
  - Total simulation time (t_max): 50 ms

Execution:
----------
1. Create a 2D cardiac tissue grid.
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
import finitewave as fw

n = 300
# create mesh
tissue = fw.CardiacTissue2D((n, n))

# set up stimulation parameters
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, 5))

# create model object and set up parameters
luo_rudy = fw.LuoRudy912D()
luo_rudy.dt = 0.01
luo_rudy.dr = 0.25
luo_rudy.t_max = 50

# add the tissue and the stim parameters to the model object
luo_rudy.cardiac_tissue = tissue
luo_rudy.stim_sequence = stim_sequence

# run the model
luo_rudy.run()

# show the potential map at the end of calculations:
plt.figure()
plt.imshow(luo_rudy.u)
plt.colorbar()
plt.show()
