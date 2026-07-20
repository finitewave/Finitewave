"""
Running the TP06 Model in 2D Cardiac Tissue
===========================================

Overview:
---------
This example demonstrates how to run a 2D simulation of the 
ten Tusscher–Panfilov 2006 (TP06) model for ventricular cardiomyocytes 
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
m = 10

stim_prepacing = fw.StimSingleCell(dt=0.01)
stim_prepacing.add_stim(n_beats=10, cycle_length=500., curr_value=20., duration=2.)

# create model object and set up parameters
tp06 = fw.TenTusscherPanfilov2006()
tp06.prepacing(stim_prepacing)


# set up stimulation parameters
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimCurrentCoord(0, 20., 2., 0, 2, 0, m))

action_pot_tracker = fw.ActionPotentialTracker(step=10)
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.node_inds = [[n//2, m//2]]
action_pot_tracker.step = 1

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(action_pot_tracker)

simulation = fw.CardiacSimulation(dt=0.01, t_max=500)
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = fw.CardiacTissue([n, m], dr=0.5)
simulation.cardiac_model = tp06
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

# run the model:
simulation.run()

# plot the action potential
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))

time = np.arange(len(tp06.u_pacing)) * simulation.dt
axs[0].plot(time, tp06.u_pacing, label="cell_50_3")
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Voltage (mV)')
axs[0].set_title('Prepacing Protocol')
axs[0].grid()

time = np.arange(len(action_pot_tracker.output)) * simulation.dt
axs[1].plot(time, action_pot_tracker.output, label="cell_50_3")
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Voltage (mV)')
axs[1].set_title('Action Potential')
axs[1].grid()
plt.tight_layout()
plt.show()