"""
Running the Barkley Model in 2D
======================================

Overview:
---------
This example demonstrates how to run a basic 2D simulation of the 
Barkley model using the Finitewave framework. 

Simulation Setup:
-----------------
- Tissue Grid: A 100×5 cardiac tissue domain.
- Stimulation:
  - A square side stimulus is applied at t = 0.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 10

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Apply a stimulus along the upper boundary to initiate excitation.
3. Set up and run the Barkley model.
4. Visualize the transmembrane potential.

"""

import matplotlib.pyplot as plt
import numpy as np
import finitewave as fw

# create a tissue of size 400x400 with cardiomycytes:
n = 100
m = 10
tissue = fw.CardiacTissueGrid([n, m], dr=0.25)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(time=0., volt_value=1,
                                           x_min=0, x_max=5,
                                           y_min=0, y_max=m))

action_pot_tracker = fw.ActionPotentialTracker()
action_pot_tracker.node_inds = [[n//2, m//2]]
action_pot_tracker.step = 1

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(action_pot_tracker)

# create model object and set up parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=6)
simulation.cardiac_model = fw.Barkley()
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

# run the model:
simulation.run()

# plot the action potential
plt.figure()
time = np.arange(len(action_pot_tracker.output)) * simulation.dt
plt.plot(time, action_pot_tracker.output, label=f"cell_{n//2}_{m//2}")
plt.legend(title='Barkley')
plt.title('Action Potential')
plt.grid()
plt.show()
