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
tissue = fw.CardiacTissue2D([n, n])

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.3
aliev_panfilov.t_max = 200

# induce spiral wave:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=0, volt_value=1, x1=0, x2=n,
                                             y1=0, y2=5))
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=50, volt_value=1, x1=n//2,
                                             x2=n, y1=0, y2=n))

# set up the tracker:
tracker_sequence = fw.TrackerSequence()
act_time_tracker = fw.LocalActivationTime2DTracker()
act_time_tracker.threshold = 0.5
act_time_tracker.step = 10
act_time_tracker.start_time = 100
act_time_tracker.end_time = 200
tracker_sequence.add_tracker(act_time_tracker)

# connect model with tissue, stim and tracker:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

# run the simulation:
aliev_panfilov.run()

# plot the activation time map:
time_bases = [150, 170]  # time bases to plot the activation time map
lats = act_time_tracker.output
print(f'Number of LATs: {len(act_time_tracker.output)}')

X, Y = np.mgrid[0:n:1, 0:n:1]

fig, axs = plt.subplots(ncols=len(time_bases), figsize=(15, 5))

if len(time_bases) == 1:
    axs = [axs]

for i, ax in enumerate(axs):
    # Select the activation times next closest to the time base
    mask = np.any(lats >= time_bases[i], axis=0)
    ids = np.argmax(lats >= time_bases[i], axis=0)
    ids = tuple((ids[mask], *np.where(mask)))

    act_time = np.full([n, n], np.nan)
    act_time[mask] = lats[ids]

    act_time_min = time_bases[i]
    act_time_max = time_bases[i] + 30

    ax.imshow(act_time,
              vmin=act_time_min,
              vmax=act_time_max,
              cmap='viridis')
    ax.set_title(f'Activation time: {time_bases[i]} ms')
    cbar = fig.colorbar(ax.images[0], ax=ax, orientation='vertical')
    cbar.set_label('Activation Time (ms)')
plt.show()