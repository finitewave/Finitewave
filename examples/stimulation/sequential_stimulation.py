"""
Sequential Multi-Type Stimulation in 2D Tissue
==============================================

Overview:
---------
This example demonstrates how to define a sequence of heterogeneous stimuli
using different stimulation classes in a single simulation using Finitewave.
Stimuli are applied from multiple locations at different times, combining
`StimVoltageCoord2D`, `StimCurrentMatrix2D`, and `StimVoltageCoord2D`.

Stimulation Setup:
------------------
1. t = 0: Voltage-based stimulation in the top-left corner (5×5 region).
2. t = 70: Matrix-based *current* stimulation (5×5 region in top-right).
3. t = 140: Voltage-based stimulation in the bottom-right corner (10×10).

Simulation Parameters:
----------------------
- Model: Aliev-Panfilov 2D
- Grid size: 200 × 200
- Time step (dt): 0.01 
- Space step (dr): 0.25
- Total simulation time: 170 

Tracking:
---------
- Animation tracker records membrane potential (`u`) every 10 steps.
- Results are saved to the "anim_data" folder and exported as a 2D animation.

Application:
------------
This setup is ideal for:
- Exploring sequential pacing protocols.
- Testing responses to multiple localized perturbations.
- Demonstrating how to combine different stimulation methods in a single run.

"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# set up the tissue:
n = 200
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
# Here we create a sequence of stimuli from different corners of a square mesh. 
# The stimulation object in the sequence can be of any type (class).
stim_sequence.add_stim(
    fw.StimVoltageCoord2D(time=0, 
                          volt_value=1.0, 
                          x1=0, x2=5, 
                          y1=0, y2=5)
)

stim_matrix = np.full([n, n], False, dtype=bool)
stim_matrix[0:5, n-5:n] = True
stim_sequence.add_stim(
    fw.StimCurrentMatrix2D(time=70, 
                           curr_value=5, 
                           duration=0.6, 
                           matrix=stim_matrix)
)

stim_sequence.add_stim(
    fw.StimVoltageCoord2D(time=140,
                          volt_value=0.5,
                          x1=n-10, x2=n,
                          y1=n-10, y2=n)
)

# set up tracker parameters:
tracker_sequence = fw.TrackerSequence()
animation_tracker = fw.Animation2DTracker()
animation_tracker.variable_name = "u"  # Specify the variable to track
animation_tracker.dir_name = "anim_data"
animation_tracker.step = 10
animation_tracker.overwrite = True  # Remove existing files in dir_name
tracker_sequence.add_tracker(animation_tracker)

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 170
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

# run the model:
aliev_panfilov.run()

# write animation and clear the snapshot folder
animation_tracker.write(shape_scale=5, clear=True, fps=60)