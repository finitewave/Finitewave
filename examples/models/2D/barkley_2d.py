"""
Running the Barkley Model in 2D
======================================

Overview:
---------
This example demonstrates how to run a basic 2D simulation of the 
Barkley model using the Finitewave framework. 

Simulation Setup:
-----------------
- Tissue Grid: A 100×5 cardiac tissue domain.
- Stimulation:
  - A square side stimulus is applied at t = 0.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 10

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Apply a stimulus along the upper boundary to initiate excitation.
3. Set up and run the Barkley model.
4. Visualize the transmembrane potential.

"""

import numpy as np
import matplotlib.pyplot as plt

import finitewave as fw

# create a tissue:
n = 100
m = 5
tissue = fw.CardiacTissue2D([n, m])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, 5, 0, m))

# create model object and set up parameters:
barkley = fw.Barkley2D()
barkley.dt = 0.01
barkley.dr = 0.25
barkley.t_max = 10
# add the tissue and the stim parameters to the model object:
barkley.cardiac_tissue = tissue
barkley.stim_sequence = stim_sequence

tracker_sequence = fw.TrackerSequence()
# add the variable tracker:
multivariable_tracker = fw.MultiVariable2DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
multivariable_tracker.cell_ind = [50, 3]
multivariable_tracker.var_list = ["u", "v"]
tracker_sequence.add_tracker(multivariable_tracker)
barkley.tracker_sequence = tracker_sequence

# run the model:
barkley.run()

# plot the action potential
plt.figure(figsize=(10, 5))

# Subplot 1: Phase plot (u vs v)
plt.subplot(1, 2, 1)
plt.plot(multivariable_tracker.output["u"], multivariable_tracker.output["v"], label="cell_50_3")
plt.legend(title='Barkley')
plt.title('Phase (u vs v)')
plt.xlabel('u')
plt.ylabel('v')
plt.grid()

# Subplot 2: Time vs u
plt.subplot(1, 2, 2)
time = np.arange(len(multivariable_tracker.output["u"])) * barkley.dt
plt.plot(time, multivariable_tracker.output["u"], label="cell_50_3")
plt.legend(title='Barkley')
plt.title('Action potential')
plt.xlabel('Time')
plt.ylabel('u')
plt.grid()

plt.show()