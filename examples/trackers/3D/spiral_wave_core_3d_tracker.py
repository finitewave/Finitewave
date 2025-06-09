
"""
Spiral Wave Core Tracking in 3D
===============================

This example demonstrates how to use the SpiralWaveCore3DTracker in Finitewave
to locate and track the core of a scroll wave (3D spiral wave) over time in 
a simulated cardiac tissue using the Aliev–Panfilov model.

Overview:
---------
- A planar wave is first initiated from the bottom of the tissue.
- A second stimulus is delivered from the left half to induce a scroll wave.
- The SpiralWaveCore3DTracker identifies the locations in the tissue where 
  phase singularities form — these correspond to the spiral wave cores.

Simulation Setup:
-----------------
- Tissue: 200×200×10 3D slab
- Time and Space:
  - Time step (dt): 0.01
  - Space step (dr): 0.25
  - Simulation duration: 150
- Stimulation:
  - t = 0 : Stimulus along the bottom edge
  - t = 31: Stimulus from the left half — creates a broken wavefront

Core Tracking:
--------------
- Threshold: 0.5 (voltage level to define wavefront)
- Start Time: 40 (after wave has developed)
- Step: 100 steps between core detections
- Output: x, y, z coordinates of scroll wave core and corresponding time points

Visualization:
--------------
The scroll wave core trajectory is visualized as a 3D scatter plot using `matplotlib`,
with the color mapped to the corresponding time of core appearance.

Applications:
-------------
- Useful for studying scroll wave dynamics and anchoring
- Helps analyze stability and drift of reentrant waves
- Can assist in identifying vulnerable tissue regions in 3D models

Note:
-----
This tracker provides sparse detection (once every `step`), and is best used
to observe long-term scroll wave motion rather than high-frequency detail.
"""


import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 200
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
aliev_panfilov.t_max = 150

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, n, 0, n//2, 0, nk))
stim_sequence.add_stim(fw.StimVoltageCoord3D(31, 1, 0, n//2, 0, n, 0, nk))

tracker_sequence = fw.TrackerSequence()
spiral_3d_tracker = fw.SpiralWaveCore3DTracker()
spiral_3d_tracker.threshold = 0.5
spiral_3d_tracker.start_time = 40
spiral_3d_tracker.step = 100
tracker_sequence.add_tracker(spiral_3d_tracker)

aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

swcore = spiral_3d_tracker.output

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(swcore['x'], swcore['y'], swcore['z'], c=swcore['time'],
           cmap='plasma', s=30)
ax.set_xlim(0, n)
ax.set_ylim(0, n)
ax.set_zlim(0, nk)
plt.show()