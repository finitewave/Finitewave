"""
Running the Barkley Model in 3D
======================================

Overview:
---------
This example demonstrates how to run a basic 3D simulation of the 
Barkley model using the Finitewave framework. 

Simulation Setup:
-----------------
- Tissue Grid: A 100×5×3 cardiac tissue domain.
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
3. Set up and run the Barkley model.
4. Visualize the transmembrane potential.

"""

import numpy as np
import matplotlib.pyplot as plt

import finitewave as fw

# create a tissue:
n = 100
m = 5
k = 3
tissue = fw.CardiacTissue3D([n, m, k])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, 5, 0, m, 0, k))

# create model object and set up parameters:
barkley = fw.Barkley3D()
barkley.dt = 0.01
barkley.dr = 0.25
barkley.t_max = 10
# add the tissue and the stim parameters to the model object:
barkley.cardiac_tissue = tissue
barkley.stim_sequence = stim_sequence

tracker_sequence = fw.TrackerSequence()
action_pot_tracker = fw.ActionPotential3DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[50, 3, 1]]
action_pot_tracker.step = 1
tracker_sequence.add_tracker(action_pot_tracker)
barkley.tracker_sequence = tracker_sequence

# run the model:
barkley.run()

# plot the action potential
plt.figure()
time = np.arange(len(action_pot_tracker.output)) * barkley.dt
plt.plot(time, action_pot_tracker.output, label="cell_50_3_1")
plt.legend(title='Barkley')
plt.title('Action Potential')
plt.grid()
plt.show()
