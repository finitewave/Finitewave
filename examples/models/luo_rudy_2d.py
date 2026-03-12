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

n = 400
m = 10

prepacing_protocol = [
    {"n_beats": 30,
     "cycle_length": 1000.,
     "stim_duration": 0.5,
     "stim_amplitude": 100.,
     "dt": 0.01},
    {"n_beats": 100,
     "cycle_length": 500.,
     "stim_duration": 0.5,
     "stim_amplitude": 100.,
     "dt": 0.01}
]
# create mesh
tissue = fw.CardiacTissueGrid((n, m), dr=0.1)

# create model object and set up parameters
luo_rudy = fw.LuoRudy91()
luo_rudy.prepacing(prepacing_protocol)
# set up stimulation parameters
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimCurrentCoord(0, 100., 1, 0, 1, 0, m))

action_pot_tracker = fw.ActionPotentialGridTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[n//2, m//2]]
action_pot_tracker.step = 1

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(action_pot_tracker)

simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.t_max = 500
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = luo_rudy
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

# run the model:
simulation.run()

# plt.plot(luo_rudy.u_pacing)
# plt.show()

# plot the action potential
plt.figure()
time = np.arange(len(action_pot_tracker.output)) * simulation.dt
plt.plot(time, action_pot_tracker.output, label="cell_50_3")
plt.legend(title='Luo-Rudy 1991')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.title('Action Potential')
plt.grid()
plt.show()