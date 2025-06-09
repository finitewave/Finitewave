"""
ActionPotential3DTracker
=========================

This example demonstrates how to use the ActionPotential3DTracker in a 3D tissue
simulation with the Aliev-Panfilov model.

Overview:
---------
The ActionPotential3DTracker allows you to monitor and record the transmembrane
potential (u) over time at specific locations within the 3D cardiac tissue.

Simulation Setup:
-----------------
- Tissue: A 3D slab of size 100×100×10 with default isotropic mesh.
- Stimulation: Planar stimulation applied at the left boundary (x ∈ [0, 3]).
- Tracking:
  - Two measurement points are selected:
    - [30, 30, 5]
    - [70, 70, 8]
  - Tracker records the value of `u` at every time step.

Execution:
----------
1. A 3D tissue is created and stimulated from one side.
2. The ActionPotential3DTracker records action potentials at the given cell
   locations throughout the simulation.
3. The recorded time series is visualized using matplotlib.

Applications:
-------------
- Useful for analyzing wave propagation, latency, and signal morphology.
- Can be used for APD measurement, restitution curve analysis, or comparing
  regional tissue responses in 3D.

Output:
-------
A plot showing transmembrane potential over time for each measurement point.
"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 100
nj = 100
nk = 10

tissue = fw.CardiacTissue3D([n, nj, nk])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, 3, 0, nj, 0, nk))

# set up tracker parameters:
tracker_sequence = fw.TrackerSequence()
action_pot_tracker = fw.ActionPotential3DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[30, 30, 5], [70, 70, 8]]
action_pot_tracker.step = 1
tracker_sequence.add_tracker(action_pot_tracker)

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilov3D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 50
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

# plot the action potential
time = np.arange(len(action_pot_tracker.output)) * aliev_panfilov.dt

plt.figure()
plt.plot(time, action_pot_tracker.output[:, 0], label="cell_30_30_5")
plt.plot(time, action_pot_tracker.output[:, 1], label="cell_70_70_8")
plt.legend(title='Aliev-Panfilov')
plt.show()