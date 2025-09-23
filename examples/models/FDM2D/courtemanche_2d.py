"""
Running the Courtemanche Model in 2D Cardiac Tissue
===========================================

Overview:
---------
This example demonstrates how to run a 2D simulation of the 
Courtemanche model for atrial cardiomyocytes 
using the Finitewave framework. 

Simulation Setup:
-----------------
- Tissue Grid: A 100×5 cardiac tissue domain.
- Stimulation:
  - A planar stimulus is applied along the top edge (rows 0 to 5) at t = 0 ms
    to initiate wave propagation.
- Time and Space Resolution:
  - Temporal step (dt): 0.01 ms
  - Spatial resolution (dr): 0.25 mm
  - Total simulation time (t_max): 500 ms

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Apply a stimulus to initiate excitation.
3. Set up and run the TP06 model.
4. Visualize the membrane potential.

"""

import numpy as np
import matplotlib.pyplot as plt
import finitewave as fw

n = 100
m = 5
# create mesh
tissue = fw.CardiacTissue2D((n, m))

# set up stimulation parameters
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, 5, 0, m))

# create model object and set up parameters
courtemanche = fw.Courtemanche2D()
courtemanche.dt = 0.01
courtemanche.dr = 0.25
courtemanche.t_max = 500

# Here, we increase g_Kur by a factor of 3 to better match physiological AP shape
# with a visible plateau and realistic repolarization.
courtemanche.gkur_coeff *= 3

# add the tissue and the stim parameters to the model object
courtemanche.cardiac_tissue = tissue
courtemanche.stim_sequence = stim_sequence

tracker_sequence = fw.TrackerSequence()
action_pot_tracker = fw.ActionPotential2DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[50, 3]]
action_pot_tracker.step = 1
tracker_sequence.add_tracker(action_pot_tracker)
courtemanche.tracker_sequence = tracker_sequence

# run the model:
courtemanche.run()

# plot the action potential
plt.figure()
time = np.arange(len(action_pot_tracker.output)) * courtemanche.dt
plt.plot(time, action_pot_tracker.output, label="cell_50_3")
plt.legend(title='Courtemanche')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.title('Action Potential')
plt.grid()
plt.show()