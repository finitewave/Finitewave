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

prepacing = fw.StimPrepacing(dt=0.01)
prepacing.add_stim(n_beats=3, cycle_length40, stim_duration=0.1, stim_amplitude=2.)
prepacing.add_stim(n_beats=3, cycle_length30, stim_duration=0.1, stim_amplitude=2.)
prepacing.add_stim(n_beats=2, cycle_length25, stim_duration=0.1, stim_amplitude=2.)

model = fw.AlievPanfilov()
model.prepacing(prepacing)

plt.plot(model.u_history)
plt.show()
