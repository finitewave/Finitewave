"""
Running the Mitchell-Schaeffer Model in 3D
======================================

Overview:
---------
This example demonstrates how to run a basic 3D simulation of the 
Mitchell-Schaeffer model using the Finitewave framework. 

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
3. Set up and run the Mitchell-Schaeffer model.
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
mitchell_schaeffer = fw.MitchellSchaeffer3D()
mitchell_schaeffer.dt = 0.01
mitchell_schaeffer.dr = 0.25
mitchell_schaeffer.t_max = 500
# add the tissue and the stim parameters to the model object:
mitchell_schaeffer.cardiac_tissue = tissue
mitchell_schaeffer.stim_sequence = stim_sequence

tracker_sequence = fw.TrackerSequence()
action_pot_tracker = fw.ActionPotential3DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[50, 3, 1]]
action_pot_tracker.step = 1
tracker_sequence.add_tracker(action_pot_tracker)
mitchell_schaeffer.tracker_sequence = tracker_sequence

# run the model:
mitchell_schaeffer.run()

# plot the action potential
plt.figure()
time = np.arange(len(action_pot_tracker.output)) * mitchell_schaeffer.dt
plt.plot(time, action_pot_tracker.output, label="cell_50_3_1")
plt.legend(title='Mitchell-Schaeffer')
plt.xlabel('Time (ms)')
plt.title('Action Potential')
plt.grid()
plt.show()
