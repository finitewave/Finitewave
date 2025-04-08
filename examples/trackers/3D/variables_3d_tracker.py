"""
Tracking State Variables in 3D Cardiac Tissue
=============================================

This example demonstrates how to use the `Variable3DTracker` and 
`MultiVariable3DTracker` classes in Finitewave to monitor the evolution of 
model state variables (e.g., transmembrane potential `u` and recovery variable `v`)
at specific cell locations within a 3D cardiac tissue model.

Overview:
---------
- The Aliev–Panfilov model is run on a 3D slab of tissue.
- Two trackers are used:
  1. `Variable3DTracker` — tracks a single variable `u` at cell (40, 40, 5).
  2. `MultiVariable3DTracker` — tracks both `u` and `v` at cell (30, 30, 5).
- A planar stimulus is applied from one side to generate an action potential.

Simulation Setup:
-----------------
- Tissue: 100×100×10 3D grid of cardiomyocytes
- Time step: 0.01
- Space step: 0.25
- Total duration: 100
- Stimulation: Small region at the front-left corner

Tracker Details:
----------------
- `Variable3DTracker` is ideal for lightweight tracking of a single variable.
- `MultiVariable3DTracker` allows simultaneous tracking of multiple state variables
  at the same spatial location.

Visualization:
--------------
The results are plotted using `matplotlib` to compare:
- The `u` values from both trackers.
- The evolution of `v` at the measurement location.

Applications:
-------------
- Useful for action potential shape analysis.
- Helps compare transmembrane dynamics across different cell locations.
- Can be used to validate ionic models or study parameter sensitivity.
"""


import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 100
nk = 10

# create tissue object:
tissue = fw.CardiacTissue3D([n, n, nk])
tissue.mesh = np.ones([n, n, nk], dtype="uint8")
tissue.add_boundaries()

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()

# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 100

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 1, 3, 1, n, 1, nk))

tracker_sequence = fw.TrackerSequence()
# add one variable tracker:
variable_tracker = fw.Variable3DTracker()
variable_tracker.var_name = "u"
variable_tracker.cell_ind = [40, 40, 5]
tracker_sequence.add_tracker(variable_tracker)

# add the multi variable tracker:
multivariable_tracker = fw.MultiVariable3DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
multivariable_tracker.cell_ind = [30, 30, 5]
multivariable_tracker.var_list = ["u", "v"]
tracker_sequence.add_tracker(multivariable_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

# plot the action potential and state variable v at the measuring point
time = np.arange(len(multivariable_tracker.output["u"])) * aliev_panfilov.dt

plt.plot(time, variable_tracker.output, label="u")
plt.plot(time, multivariable_tracker.output["u"], label="u")
plt.plot(time, multivariable_tracker.output["v"], label="v")
plt.legend(title=aliev_panfilov.__class__.__name__)
plt.show()