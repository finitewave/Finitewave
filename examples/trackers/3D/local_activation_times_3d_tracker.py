"""
Local Activation Time in 3D
===========================

This example demonstrates how to use the LocalActivationTime3DTracker to track 
the local activation times in a 3D cardiac tissue slab using the Aliev–Panfilov model.

Overview:
---------
The LocalActivationTime3DTracker records activation times at each node when the
membrane potential crosses a defined threshold. Unlike standard activation time 
trackers, it can store multiple activations (e.g., from reentry or spiral waves)
and enables detailed temporal analysis of wavefront propagation.

Simulation Setup:
-----------------
- Tissue: A 3D slab of size 200×200×10.
- Stimulation:
  - First stimulus: a planar front at y=0–5, applied at t=0.
  - Second stimulus: half of the domain (x=n/2 to n), applied at t=50.
- Model:
  - Aliev–Panfilov in 3D with dt = 0.01 and dr = 0.3 units.
  - Total simulation time: 200.
- Tracker:
  - `LocalActivationTime3DTracker` activated from t=100 to t=200.
  - Records activation times every 10 steps (step=10).
  - Activation threshold set to 0.5.

Visualization:
--------------
- Two time points (150 and 170) are visualized.
- For each, a 3D scatter plot shows all nodes activated at or after the given time.
- Activation time is color-coded using a viridis colormap.

Applications:
-------------
- Visualization of reentrant waves in 3D.
- Analysis of wavefront timing and conduction delays.
- Studying effects of geometry, fibrosis, or heterogeneity on activation dynamics.

Output:
-------
- Two 3D plots showing activation times at specified time bases.
- Printed number of LATs (activation events) recorded by the tracker.
"""


import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 200
nk = 10
tissue = fw.CardiacTissue3D([n, n, nk])

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.3
aliev_panfilov.t_max = 200

# induce spiral wave:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(time=0, volt_value=1, x1=0, x2=n,
                                             y1=0, y2=5, z1=0, z2=nk))
stim_sequence.add_stim(fw.StimVoltageCoord3D(time=50, volt_value=1, x1=n//2,
                                             x2=n, y1=0, y2=n, z1=0, z2=nk))

# set up the tracker:
tracker_sequence = fw.TrackerSequence()
act_time_tracker = fw.LocalActivationTime3DTracker()
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

fig = plt.figure(figsize=(15, 5))

for i, time_base in enumerate(time_bases):
    ax = fig.add_subplot(1, len(time_bases), i + 1, projection='3d')

    # Select the activation times next closest to the time base
    mask = np.any(lats >= time_base, axis=0)
    ids = np.argmax(lats >= time_base, axis=0)
    ids = tuple((ids[mask], *np.where(mask)))

    act_time = np.full([n, n, nk], np.nan)
    act_time[mask] = lats[ids]

    act_time_min = time_base
    act_time_max = time_base + 30

    # Create a 3D scatter plot
    x, y, z = np.where(~np.isnan(act_time))
    values = act_time[~np.isnan(act_time)]

    scatter = ax.scatter(x, y, z, c=values, cmap='viridis', vmin=act_time_min, vmax=act_time_max)
    ax.set_title(f'Activation time: {time_base} ms')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.1)
    cbar.set_label('Activation Time (ms)')

plt.tight_layout()
plt.show()