
"""
Tracking Action Potentials in 2D Cardiac Tissue
===============================================

Overview:
---------
This example demonstrates how to track action potentials at specific 
cell locations in a 2D cardiac tissue simulation using the 
ActionPotential2DTracker class in Finitewave. Action potential tracking 
is crucial for analyzing electrophysiological responses at different 
tissue points.

Simulation Setup:
-----------------
- Tissue Grid: A 100×100 cardiac tissue domain.
- Stimulation:
  - A left-side stimulus is applied at time t = 0.
  - The excitation wave propagates across the tissue.
- Action Potential Tracking:
  - Action potentials are recorded at two specific cells:  
    - Cell at (30, 30)
    - Cell at (70, 70)
  - Sampling step: Every time step (1 ms).
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 50

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Apply stimulation at the left boundary.
3. Set up an action potential tracker:
   - The tracker records the membrane potential over time at specified 
     cell indices.
4. Run the Aliev-Panfilov model to simulate wave propagation.
5. Extract and visualize action potential waveforms.

Application:
------------
Tracking action potentials is useful for:
- Studying cardiac excitability at different spatial locations.
- Comparing action potential durations across various tissue points.
- Analyzing arrhythmias or conduction abnormalities in excitable media.

Visualization:
--------------
The action potentials recorded at the selected cells are plotted over time 
using matplotlib. The graph shows the voltage dynamics of the 
excited regions.

"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# create a mesh of cardiomyocytes (elems = 1):
n = 100
m = 100
tissue = fw.CardiacTissue2D([m, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, 3, 0, n))

# set up tracker parameters:
tracker_sequence = fw.TrackerSequence()
action_pot_tracker = fw.ActionPotential2DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[30, 30], [70, 70]]
action_pot_tracker.step = 1
tracker_sequence.add_tracker(action_pot_tracker)

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilov2D()
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
plt.plot(time, action_pot_tracker.output[:, 0], label="cell_30_30")
plt.plot(time, action_pot_tracker.output[:, 1], label="cell_70_70")
plt.legend(title='Aliev-Panfilov')
plt.show()
