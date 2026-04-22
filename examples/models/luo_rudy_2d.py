"""
Running the Luo-Rudy 1991 Model in 2D Cardiac Tissue
====================================================

Overview:
---------
This example demonstrates how to run a 2D simulation of the 
Luo-Rudy 1991 ventricular action potential model using the Finitewave framework.

Simulation Setup:
-----------------
- Tissue Grid: A 100×5 cardiac tissue domain.
- Stimulation:
  - A planar stimulus is applied along the top edge of the domain at t = 0 ms
    to initiate wavefront propagation.
- Time and Space Resolution:
  - Temporal step (dt): 0.01 ms
  - Spatial resolution (dr): 0.25 mm
  - Total simulation time (t_max): 500 ms

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Apply a stimulus along the upper boundary to initiate excitation.
3. Set up and run the Luo-Rudy 1991 model.
4. Visualize the transmembrane potential.

"""

import numpy as np
import matplotlib.pyplot as plt
import finitewave as fw

n = 100
m = 100
dt = 0.01
t_max = 500

# create mesh
tissue = fw.CardiacTissueGrid((n, m), dr=0.1)

# create model object and set up parameters
stim_prepacing = fw.StimPrepacing(dt)
stim_prepacing.add_stim(n_beats=10, cycle_length=1000., curr_value=100, duration=0.5)

luo_rudy = fw.LuoRudy91()
luo_rudy.prepacing(stim_prepacing)
# set up stimulation parameters
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimCurrentCoord(0, 100., 1, 0, 1, 0, m))

action_pot_tracker = fw.ActionPotentialTracker(step=100)
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.node_inds = [[n//2, m//2]]
action_pot_tracker.step = 1

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(action_pot_tracker)

simulation = fw.CardiacSimulation(dt=dt, t_max=t_max, backend="jax")
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = luo_rudy
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

# run the model:
simulation.run()

# plot the action potential
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))

time = np.arange(len(luo_rudy.u_pacing)) * dt
axs[0].plot(time, luo_rudy.u_pacing, label=f"cell_{n//2}_{m//2}")
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Voltage (mV)')
axs[0].set_title('Prepacing Protocol')
axs[0].grid()

axs[1].plot(action_pot_tracker.tracking_times, action_pot_tracker.output, label=f"cell_{n//2}_{m//2}")
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Voltage (mV)')
axs[1].set_title('Action Potential')
axs[1].grid()
plt.tight_layout()
plt.legend(title='Luo-Rudy 1991')
plt.show()