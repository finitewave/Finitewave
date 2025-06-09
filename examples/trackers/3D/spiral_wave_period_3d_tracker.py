"""
Wave Period in 3D Tissue
========================

This example demonstrates how to use the Period3DTracker in Finitewave to measure 
the wave period at specific locations in a 3D cardiac tissue simulation using 
the Aliev–Panfilov model.

Overview:
---------
The Period3DTracker detects threshold crossings (e.g., wave upstrokes) at 
specified cells to estimate the local activation period. This is useful for 
analyzing rhythm stability in sustained wave activity such as spiral or scroll waves.

Simulation Setup:
-----------------
- Tissue Size: 100×100×10
- Initial Conditions: Fully excitable tissue with no fibrosis
- Boundary Handling: No-flux boundaries using `add_boundaries()`
- Stimulation:
  - First planar stimulus at t = 0, applied to lower half of Y domain
  - Second planar stimulus at t = 31, applied to left half of X domain
  - This induces spiral-like propagation dynamics

Period Measurement:
-------------------
- Tracker: Period3DTracker
- Target Cells: 7 manually selected positions within the mid-slice (z = 5)
- Threshold: 0.5 (voltage level for upstroke detection)
- Start Time: 100 (to allow initiation to settle)
- Step: 10 (check voltage every 10 steps)

Output:
-------
- Mean and standard deviation of measured periods per cell
- A matplotlib errorbar plot shows variability across spatial locations

Application:
------------
- Useful for scroll/spiral wave analysis
- Can help detect regions with rhythm instability or alternans
- Supports investigation of how geometry or fibrosis affects pacing regularity

Note:
-----
For full local activation time maps and wavefront tracking, consider using 
`LocalActivationTime3DTracker`.
"""


import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 100
nk = 10

tissue = fw.CardiacTissue3D([n, n, nk])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n, nk], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 300

# induce spiral wave:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, n, 0, n//2, 0, nk))
stim_sequence.add_stim(fw.StimVoltageCoord3D(31, 1, 0, n//2, 0, n, 0, nk))

tracker_sequence = fw.TrackerSequence()
period_tracker = fw.Period3DTracker()
positions = np.array([[1, 1, 5],
                      [5, 5, 5],
                      [7, 3, 5],
                      [9, 1, 5],
                      [50, 50, 5],
                      [75, 3, 5],
                      [50, 75, 5]])
period_tracker.cell_ind = positions
period_tracker.threshold = 0.5
period_tracker.start_time = 100
period_tracker.step = 10
tracker_sequence.add_tracker(period_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

# get the wave period as a pandas Series with the cell index as the index:
periods = period_tracker.output

# plot the wave period:
plt.figure()
plt.errorbar(range(len(positions)),
             periods.apply(lambda x: x.mean()),
             yerr=periods.apply(lambda x: x.std()),
             fmt='o')
plt.xticks(range(len(positions)),
           [f'({x[0]}, {x[1]}, {x[2]})' for x in positions],
           rotation=45)
plt.xlabel('Cell Index')
plt.ylabel('Period')
plt.title('Wave period')
plt.tight_layout()
plt.show()