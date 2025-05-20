"""
Animation3DTracker Example
==========================

This example demonstrates how to use the Animation3DTracker to generate a
visualization of transmembrane potential (u) over time in a 3D cardiac tissue
simulation using the Aliev-Panfilov model.

Overview:
---------
The tracker captures snapshots of the selected variable during the simulation
and later compiles them into an animation (e.g. .mp4 video).

Simulation Setup:
-----------------
- Tissue: A 3D slab of size 100×100×10.
- Stimulation:
  - First wave is initiated from the lower half of the tissue at t = 0.
  - Second wave is initiated from the left half at t = 31 to create 
    wavefront interactions.
- Tracking:
  - The transmembrane potential (`u`) is recorded every 10 steps.
  - Snapshots are stored in the folder `anim_data` and compiled into a .mp4.

Execution:
----------
1. Simulate wave propagation using the Aliev-Panfilov model.
2. Save snapshots of `u` at regular intervals.
3. Compile snapshots into an animation after the simulation.

Applications:
-------------
- Useful for visualizing wave dynamics in 3D, such as propagation, collision,
  or reentry.
- Supports model validation, presentation, and educational use.

Output:
-------
An `.mp4` animation file in the `anim_data` folder, showing how `u` evolves
over time in the 3D domain.
"""

import numpy as np

import finitewave as fw

# set up the tissue:
n = 100
nk = 10
tissue = fw.CardiacTissue3D([n, n, nk])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, n, 0, n//2, 0, nk))
stim_sequence.add_stim(fw.StimVoltageCoord3D(31, 1, 0, n//2, 0, n, 0, nk))

# set up tracker parameters:
tracker_sequence = fw.TrackerSequence()
animation_tracker = fw.Animation3DTracker()
animation_tracker.variable_name = "u"  # Specify the variable to track
animation_tracker.dir_name = "anim_data"
animation_tracker.step = 10
animation_tracker.overwrite = True  # Remove existing files in dir_name
tracker_sequence.add_tracker(animation_tracker)

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 100
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence
# run the model:
aliev_panfilov.run()

# write animation and clear the snapshot folder
animation_tracker.write(format='mp4', framerate=10, quality=9,
                        clear=True, clim=[0, 1]) # !Note: for ionic models use clim=[-90, 40] or similar to show the activity correctly