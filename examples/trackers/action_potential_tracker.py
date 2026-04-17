
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
- Tissue Grid: A 100×10 cardiac tissue domain with a spatial resolution of dr = 0.25.
- Stimulation:
  - A left-side stimulus is applied at time t = 0.
  - The excitation wave propagates across the tissue.
- Action Potential Tracking:
  - Action potentials are recorded at two specific cells:  
    - Cell at (30, 5)
    - Cell at (70, 5)
  - Sampling step: Every time step.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
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
m = 10
tissue = fw.CardiacTissueGrid([n, m], dr=0.25)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimCurrentCoord(0, 5, 0.1, 0, 3, 0, m))

# set up tracker parameters:
node_inds = [[0, 5], [70, 5]]
action_pot_tracker = fw.ActionPotentialTracker(node_inds)

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(action_pot_tracker)

# set up simulation parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=50)
simulation.cardiac_model = fw.AlievPanfilov(memory_save=False)
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

# run the model:
simulation.run()

# plot the action potential
time = action_pot_tracker.tracking_times
act_pot = action_pot_tracker.act_pot
u = simulation.cardiac_model.u

fig, axs = plt.subplots(ncols=2, width_ratios=[0.3, 1])

axs[0].imshow(u, cmap="RdBu_r", origin="lower")
axs[0].scatter(np.array(action_pot_tracker.node_inds)[:, 1],
               np.array(action_pot_tracker.node_inds)[:, 0],
               c=["tab:blue", "tab:orange"],
               label='Tracked Nodes')

for i, (x, y) in enumerate(node_inds):
    axs[1].plot(time, act_pot[:, i], label=f"Node [{x}, {y}]")
axs[1].legend(title='Aliev-Panfilov Model')
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Membrane Potential (mV)')
plt.show()
