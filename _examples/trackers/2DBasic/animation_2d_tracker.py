
"""
# Animation in 2D

This example demonstrates how to use the Animation2DTracker to build an
animation of the action potential in 2D.
"""

import numpy as np
import finitewave as fw

# number of nodes on the side
n = 100
tissue = fw.CardiacTissue2D([n, n])
# add some 30% of fibrosis
tissue.mesh[np.random.random([n, n]) < 0.3] = 2
# ensure that the boundaries are set correctly:
tissue.add_boundaries()

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, 5))

# set up tracker parameters:
tracker_sequence = fw.TrackerSequence()
animation_tracker = fw.Animation2DTracker()
animation_tracker.variable_name = "u"  # Specify the variable to track
animation_tracker.dir_name = "anim_data"
animation_tracker.step = 10
animation_tracker.overwrite = True  # Remove existing files in dir_name
tracker_sequence.add_tracker(animation_tracker)

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 50
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence
# run the model:
aliev_panfilov.run()

# write animation and clear the snapshot folder
animation_tracker.write(shape_scale=5, clear=True, fps=30)
