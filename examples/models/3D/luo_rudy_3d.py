"""
Running the Luo-Rudy 1991 Model in 3D Cardiac Tissue
====================================================

Overview:
---------
This example demonstrates how to run a 3D simulation of the 
Luo-Rudy 1991 ventricular action potential model using the Finitewave framework.

Simulation Setup:
-----------------
- Tissue Grid: A 100×5×3 cardiac tissue domain.
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
4. Visualize the transmembrane potential.

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
luo_rudy = fw.LuoRudy913D()
luo_rudy.dt = 0.01
luo_rudy.dr = 0.25
luo_rudy.t_max = 500

# add the tissue and the stim parameters to the model object
luo_rudy.cardiac_tissue = tissue
luo_rudy.stim_sequence = stim_sequence

tracker_sequence = fw.TrackerSequence()
action_pot_tracker = fw.ActionPotential3DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[50, 3, 1]]
action_pot_tracker.step = 1
tracker_sequence.add_tracker(action_pot_tracker)
luo_rudy.tracker_sequence = tracker_sequence

# run the model:
luo_rudy.run()

# plot the action potential
plt.figure()
time = np.arange(len(action_pot_tracker.output)) * luo_rudy.dt
plt.plot(time, action_pot_tracker.output, label="cell_50_3_1")
plt.legend(title='Luo-Rudy 1991')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.title('Action Potential')
plt.grid()
plt.show()