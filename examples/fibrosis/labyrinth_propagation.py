"""
Simulation in Complex Labyrinth-Like Geometry
=============================================

This example demonstrates wave propagation through a 2D cardiac tissue
with a custom-designed labyrinth-like structure. The geometry is created
manually by setting up regions of obstacles (non-conductive) within a
conductive domain. The resulting structure mimics pathways similar to
fibrotic maze-like or post-surgical scarred tissue.

Wavefront propagation is visualized using Finitewave’s Animation2DTracker,
and the result shows how the wave navigates through the complex network
of narrow channels and dead-ends.

Setup:
------
- Tissue size: 300 × 300
- Geometry:
    • Obstacles are placed in alternating vertical bands
    • Bands are offset to form a labyrinth pattern
    • `tissue.mesh` uses 1 (myocytes) and 0 (obstacles)
- Stimulus:
    • A short planar stimulus applied to a small strip on the left side
    • Time: t = 0 ms
- Model:
    • Aliev-Panfilov 2D model
    • Total time: 200 ms
- Visualization:
    • Voltage (`u`) is tracked every 10 steps
    • Animation frames are saved and compiled to visualize dynamics

Output:
-------
To visualize the result, refer to the generated animation (e.g., 
`complex_geometry.mp4`) showing how wavefronts propagate within the complex structure.

"""

import matplotlib.pyplot as plt
import numpy as np
import shutil

import finitewave as fw

# number of nodes on the side
n = 300

tissue = fw.CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
for i in range(0, 40, 5):
    if i%10 == 0:
        tissue.mesh[10*i:10*(i+3), :250] = 0
    else:
        tissue.mesh[10*i:10*(i+3), 50:] = 0

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 200

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, int(n*0.03),
                                                 0, n))

tracker_sequence = fw.TrackerSequence()
animation_tracker = fw.Animation2DTracker()
animation_tracker.variable_name = "u"  # Specify the variable to track
animation_tracker.dir_name = "anim_data"
animation_tracker.step = 10
animation_tracker.overwrite = True  # Remove existing files in dir_name
tracker_sequence.add_tracker(animation_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

# write animation and clear the snapshot folder
animation_tracker.write(shape_scale=5, clear=True, fps=30)
