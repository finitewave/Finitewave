"""
Tracking Activation Time in 2D Cardiac Tissue
=============================================

Overview:
---------
This example demonstrates how to track activation times during a 
2D cardiac tissue simulation using the ActivationTime2DTracker 
class in Finitewave. Activation time tracking helps analyze the propagation 
of electrical waves and conduction delays in excitable media.

Simulation Setup:
-----------------
- Tissue Grid: A 200×200 cardiac tissue domain.
- Stimulation:
  - A left-side stimulus is applied at time t = 0.
  - The excitation propagates across the tissue.
- Activation Time Tracking:
  - Threshold: 0.5 (membrane potential value used to define activation).
  - Sampling interval: Every 100 steps.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 50

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Apply stimulation along the left boundary to initiate wave propagation.
3. Set up an activation time tracker:
   - The tracker records the time of first activation for each tissue element.
4. Run the Aliev-Panfilov model to simulate wave dynamics.
5. Extract and visualize the activation time map.

Application:
------------
Tracking activation times is useful for:
- Analyzing conduction velocity in cardiac tissue.
- Detecting conduction blocks or delays in pathological conditions.
- Comparing different tissue properties (e.g., isotropic vs. anisotropic).

Visualization:
--------------
The activation time map is displayed using matplotlib, with a color-coded 
representation of activation delays across the tissue.

"""

import matplotlib.pyplot as plt

import finitewave as fw

# create a mesh of cardiomyocytes (elems = 1):
n = 200
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=0, volt_value=1,
                                             x1=0, x2=3, y1=0, y2=n))

# set up tracker parameters:
tracker_sequence = fw.TrackerSequence()
act_time_tracker = fw.ActivationTime2DTracker()
act_time_tracker.threshold = 0.5
act_time_tracker.step = 100  # calculate activation time every 100 steps
tracker_sequence.add_tracker(act_time_tracker)

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 50
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

# plot the activation time map
plt.imshow(act_time_tracker.output, cmap="viridis")
cbar = plt.colorbar()
cbar.ax.set_ylabel('Time (model units)', rotation=270, labelpad=15)
plt.title("Activation time map")
plt.show()
