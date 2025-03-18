
#
# Sequential stimulation that can be used for high pacing protocol simulation.
# In this example we stimulate the tissue at 0, 30, 60, 90 time points with planar wave. 
# Here se used Current stimulation 
# Check the stim_sequence.mp4 to see the result.
# 

import numpy as np
import shutil

import finitewave as fw

# number of nodes on the side
n = 400

tissue = fw.CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n])
tissue.add_boundaries()
# add numeric method stencil for weights computations
# IsotropicStencil is default stencil and will be ised if no stencil was specified
tissue.stencil = fw.IsotropicStencil2D()

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 100

# set up stimulation parameters:
stim_sequence = fw.StimSequence()

for t in [0, 30, 60, 90]: # time sequence (time, curr value, curr stim time, rectangular area)
    stim_sequence.add_stim(fw.StimCurrentCoord2D(t, 3, 0.1, 0, int(n*0.03),
                                                 0, n))
    
tracker_sequence = fw.TrackerSequence()
# add action potential tracker
animation_tracker = fw.Animation2DTracker()
# We want to write the animation for the voltage variable. Use string value
# to specify the required array.anim_data
animation_tracker.target_array = "u"
# Folder name:
animation_tracker.dir_name = "anim_data"
animation_tracker.step = 1
tracker_sequence.add_tracker(animation_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue   = tissue
aliev_panfilov.stim_sequence    = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

animation_builder = fw.Animation2DBuilder()
animation_builder.dir_name = "anim_data"
animation_builder.write(shape_scale=5, clear=True, fps=30)

# remove the snapshots folder:
shutil.rmtree("anim_data")
