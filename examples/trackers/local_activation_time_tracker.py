"""
Tracking Local Activation Time in 2D Cardiac Tissue
===================================================

Overview:
---------
This example demonstrates how to use the `LocalActivationTime2DTracker` to 
track multiple local activation events over time in a 2D cardiac tissue 
simulation using the Aliev-Panfilov model. Unlike `ActivationTime2DTracker`, 
which stores only the first activation time per cell, this tracker captures 
all threshold crossings during a specified time window.

Simulation Setup:
-----------------
- Tissue Grid: A 200×200 cardiac tissue domain.
- Spiral Wave Initiation:
  - First stimulus at t = 0 along the top edge.
  - Second stimulus at t = 50 applied to the right half of the tissue.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.3
  - Total simulation time (t_max): 200

Local Activation Time Tracking:
-------------------------------
- Threshold: 0.5 (value of `u` used to detect activation).
- Records all threshold crossings per cell during:
  - `start_time = 100`
  - `end_time = 200`
- Data is recorded every `step = 10` simulation steps.
- The tracker outputs a 3D array (num_events, x, y) with activation times.

Execution:
----------
1. Set up a 2D tissue grid and stimulation pattern to induce spiral activity.
2. Configure the `LocalActivationTime2DTracker`.
3. Run the simulation using the Aliev-Panfilov model.
4. Extract and visualize activation maps for selected time points.

Application:
------------
- Ideal for analyzing wave reentry, rotation, or drift.
- Helps evaluate activation frequency and reactivation patterns.
- Useful in quantifying arrhythmogenic behavior over time.

Visualization:
--------------
Activation time maps are plotted for selected reference time bases (e.g. 150, 170), 
showing the most recent activation at each location relative to that time base.

Output:
-------
A set of color-mapped images visualizing activation wavefronts at different times, 
with all threshold-crossing events taken into account.

"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 200
tissue = fw.CardiacTissue([n, n], dr=0.3)

model = fw.AlievPanfilov()

# induce spiral wave:
stim_sequence = fw.StimS1S2Cross(tissue, s1_time=0, s2_time=31, voltage_value=1)

# set up the tracker:
lat_tracker = fw.LocalActivationTimeTracker(threshold=0.5, step=1, 
                                            start_time=100, end_time=200)

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(lat_tracker)

# set up the simulation:
simulation = fw.CardiacSimulation(dt=0.01, t_max=200, backend="numba")
simulation.cardiac_tissue = tissue
simulation.cardiac_model = model
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

# run the simulation:
simulation.run()

# plot the activation time map:
time_bases = [100, 150 ]  # time bases to plot the activation time map
print(f'Number of LATs: {len(lat_tracker.output)}')

fig, axs = plt.subplots(ncols=len(time_bases), figsize=(5 * len(time_bases), 5))

if len(time_bases) == 1:
    axs = [axs]

for i, ax in enumerate(axs):
    time_min = time_bases[i]
    time_max = time_bases[i] + 28

    lat_map = lat_tracker.activation_map(time_min, time_max).reshape(n, n)

    ax.imshow(lat_map, cmap='hsv', origin='lower')
    ax.set_title(f'LAT after {time_bases[i]} time units')
    cbar = fig.colorbar(ax.images[0], ax=ax, orientation='vertical')
    cbar.set_label('LAT (time units)')

# plt.tight_layout()
plt.show()
