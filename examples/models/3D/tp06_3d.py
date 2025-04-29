"""
Running the TP06 Model in 3D Cardiac Tissue
===========================================

Overview:
---------
This example demonstrates how to run a 3D simulation of the 
ten Tusscher–Panfilov 2006 (TP06) model for ventricular cardiomyocytes 
using the Finitewave framework. 

Simulation Setup:
-----------------
- Tissue Grid: A 100×5×3 cardiac tissue domain.
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
4. Visualize the membrane potential.

"""

import numpy as np
import matplotlib.pyplot as plt
import finitewave as fw

n = 100
m = 5
k = 3
# create mesh
tissue = fw.CardiacTissue3D((n, m, k))

# set up stimulation parameters
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, 5, 0, m, 0, k))

# create model object and set up parameters
tp06 = fw.TP063D()
tp06.dt = 0.01
tp06.dr = 0.25
tp06.t_max = 500

# add the tissue and the stim parameters to the model object
tp06.cardiac_tissue = tissue
tp06.stim_sequence = stim_sequence

tracker_sequence = fw.TrackerSequence()
action_pot_tracker = fw.ActionPotential3DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[50, 3, 1]]
action_pot_tracker.step = 1
tracker_sequence.add_tracker(action_pot_tracker)
tp06.tracker_sequence = tracker_sequence

# run the model:
tp06.run()

# plot the action potential
plt.figure()
time = np.arange(len(action_pot_tracker.output)) * tp06.dt
plt.plot(time, action_pot_tracker.output, label="cell_50_3_1")
plt.legend(title='Ten Tusscher-Panfilov 2006')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.title('Action Potential')
plt.grid()
plt.show()