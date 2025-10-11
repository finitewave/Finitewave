"""
Running the Aliev-Panfilov Model in 2D
======================================

Overview:
---------
This example demonstrates how to run a basic 2D simulation of the 
Aliev-Panfilov model using the Finitewave framework. 

Simulation Setup:
-----------------
- Tissue Grid: A 100×5 cardiac tissue domain.
- Stimulation:
  - A square side stimulus is applied at t = 0.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 50

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Apply a stimulus along the upper boundary to initiate excitation.
3. Set up and run the Aliev-Panfilov model.
4. Visualize the transmembrane potential.

"""

import numpy as np
import matplotlib.pyplot as plt

import finitewave.gridywave as fw

# create a tissue:
n = 400
m = 400

tissue = fw.CardiacTissueGrid([n, m])
tissue.mesh[np.random.rand(n, m) < 0.3] = 2  # introduce some inexcitable regions

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageGridCoord(0, 1, 0, 5, 0, m))

action_pot_tracker = fw.ActionPotentialGridTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[50, 3]]
action_pot_tracker.step = 1

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(action_pot_tracker)

simulation = fw.CardiacGridSimulation()
simulation.dt = 0.01
simulation.dr = 0.25
simulation.t_max = 300
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.AlievPanfilov()
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

# run the model:
simulation.run()

# visualize the results:
plt.imshow(simulation.cardiac_model.u, cmap='jet', origin='lower')
plt.colorbar(label='Transmembrane Potential (u)')
plt.title('Aliev-Panfilov Model - Transmembrane Potential')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# plot the action potential
# plt.figure()
# time = np.arange(len(action_pot_tracker.output)) * simulation.dt
# plt.plot(time, action_pot_tracker.output, label="cell_50_3")
# plt.legend(title='Aliev-Panfilov')
# plt.title('Action Potential')
# plt.grid()
# plt.show()
