"""
Tracking State Variables in 2D Cardiac Tissue
=============================================

Overview:
---------
This example demonstrates how to use the `MultiVariable2DTracker` to record 
the values of multiple state variables (such as `u` and `v`) at a specific 
cell in a 2D cardiac tissue simulation using the Aliev-Panfilov model.

Simulation Setup:
-----------------
- Tissue Grid: A 100×100 cardiac tissue domain.
- Stimulation:
  - A stimulus is applied to the left edge of the domain at t = 0 to initiate wave propagation.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 100

State Variable Tracking:
------------------------
- The `MultiVariable2DTracker` is used to track both:
  - `u`: Transmembrane potential
  - `v`: Recovery variable
- Tracking location is set via `cell_ind = [30, 30]`.
- Variable values are recorded at every time step.

Execution:
----------
1. Set up a 2D cardiac tissue grid and stimulation pattern.
2. Attach the `MultiVariable2DTracker` to record `u` and `v` at one node.
3. Run the simulation using the Aliev-Panfilov model.
4. Plot the recorded values over time to analyze the local action potential dynamics.

Application:
------------
- Useful for analyzing the temporal dynamics of variables at specific tissue points.
- Can help validate model behavior or compare different cell locations.
- Ideal for creating time series data for further signal analysis or machine learning tasks.

Output:
-------
The resulting plot shows the evolution of both `u` and `v` at the selected cell, 
providing insight into the local electrophysiological response to stimulation.

"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 100
tissue = fw.CardiacTissue2D([n, n])

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()

# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 100

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, 3, 0, n))

tracker_sequence = fw.TrackerSequence()

# add the variable tracker:
multivariable_tracker = fw.MultiVariable2DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
multivariable_tracker.cell_ind = [30, 30]
multivariable_tracker.var_list = ["u", "v"]
tracker_sequence.add_tracker(multivariable_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

# plot the action potential and state variable v at the measuring point
time = np.arange(len(multivariable_tracker.output["u"])) * aliev_panfilov.dt

plt.plot(time, multivariable_tracker.output["u"], label="u")
plt.plot(time, multivariable_tracker.output["v"], label="v")
plt.legend(title=aliev_panfilov.__class__.__name__)
plt.show()