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

import matplotlib.pyplot as plt
import numpy as np
import finitewave as fw

# create a tissue of size 400x400 with cardiomycytes:
n = 400
tissue = fw.CardiacTissueGrid([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(time=0, volt_value=1,
                                           x_min=n//2 - 3, x_max=n//2 + 3,
                                           y_min=n//2 - 3, y_max=n//2 + 3))

action_pot_tracker = fw.ActionPotentialGridTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[50, 3]]
action_pot_tracker.step = 1

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(action_pot_tracker)

# create model object and set up parameters:
simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.dr = 0.25
simulation.t_max = 30
simulation.cardiac_model = fw.AlievPanfilov()
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence

# run the model:
simulation.run()

# plot the action potential
plt.figure()
time = np.arange(len(action_pot_tracker.output)) * simulation.dt
plt.plot(time, action_pot_tracker.output, label="cell_50_3")
plt.legend(title='Aliev-Panfilov')
plt.title('Action Potential')
plt.grid()
plt.show()
