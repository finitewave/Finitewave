"""
Creating an Animation of Action Potential in 2D
===============================================

Overview:
---------
This example demonstrates how to use the `Animation2DTracker` to generate an 
animation of the action potential (or any other variablse you choose) in a 2D cardiac tissue simulation. 
The animation is saved as a sequence of frames and can be exported as a video or GIF.

Simulation Setup:
-----------------
- Tissue Grid: A 200×200 cardiac tissue domain.
- Stimulation:
  - First stimulus is applied to the bottom half of the domain at t = 0.
  - Second stimulus is applied to the left half at t = 31 to initiate wave break and spiral formation.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 100

Animation Tracker:
------------------
- Tracks the transmembrane potential `u` during the simulation.
- Records a frame every 10 steps (`step = 10`).
- Frames are saved into the `anim_data/` directory.
- Existing data in the directory will be overwritten.
- After the simulation, `write()` is called to render the animation:
  - `shape_scale`: Zoom factor for each frame.
  - `clear=True`: Deletes all raw frame data after animation is generated.
  - `fps=30`: Frames per second for the output video.

Execution:
----------
1. Set up cardiac tissue and stimulation sequence.
2. Attach `Animation2DTracker` to the tracker sequence.
3. Run the Aliev-Panfilov model with configured simulation and tracking.
4. Call `write()` to generate and optionally clean up the animation.

Application:
------------
- Useful for visualizing wave propagation and reentry.
- Can be used in presentations, publications, or model comparisons.
- Helps in debugging wave dynamics and understanding tissue behavior.

Output:
-------
The animation is written to disk in the specified folder (`anim_data`). 
It shows the evolution of the transmembrane potential over time in the tissue.

"""

import numpy as np

import finitewave as fw

# set up the tissue:
n = 200
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, n//2))
stim_sequence.add_stim(fw.StimVoltageCoord2D(31, 1, 0, n//2, 0, n))

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
aliev_panfilov.t_max = 100
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence
# run the model:
aliev_panfilov.run()

# write animation and clear the snapshot folder
animation_tracker.write(shape_scale=5, clear=True, fps=30)