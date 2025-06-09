
"""
Tracking Spiral Wave Core in 2D Cardiac Tissue
==============================================

Overview:
---------
This example demonstrates how to use the SpiralWaveCore2DTracker to track 
the core of a spiral wave in a 2D cardiac tissue simulation. Spiral 
waves are essential phenomena in cardiac electrophysiology and are closely 
related to reentrant arrhythmias.

Simulation Setup:
-----------------
- Tissue Grid: A 200×200 cardiac tissue domain.
- Spiral Wave Initiation:
  - A first stimulus excites the lower half of the tissue at t = 0.
  - A second stimulus is applied to the left half at t = 31, 
    breaking the wavefront and initiating spiral wave formation.
- Spiral Core Tracking:
  - Threshold: 0.5 (voltage level used to detect the wave core).
  - Tracking start time: 50 (after wave formation).
  - Recording interval: Every 100 steps.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 300

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Apply two sequential stimulations to induce a spiral wave.
3. Set up a spiral wave core tracker:
   - Tracks the movement of the wave’s center over time.
4. Run the Aliev-Panfilov model to simulate wave dynamics.
5. Extract and visualize the spiral wave trajectory.

Application:
------------
Tracking the spiral wave core is useful for:
- Analyzing reentrant arrhythmias and spiral wave stability.
- Studying spiral wave drift and anchoring in different tissue conditions.
- Testing anti-arrhythmic strategies by analyzing wave behavior.

Visualization:
--------------
The spiral wave trajectory is plotted over the final membrane potential 
distribution using matplotlib, showing how the wave core moves over time.

"""

import matplotlib.pyplot as plt

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
sw_core_tracker = fw.SpiralWaveCore2DTracker()
sw_core_tracker.threshold = 0.5
sw_core_tracker.start_time = 50
sw_core_tracker.step = 100  # Record the spiral wave core every 1 time unit
tracker_sequence.add_tracker(sw_core_tracker)

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 300
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

sw_core = sw_core_tracker.output

# plot the spiral wave trajectory:
plt.imshow(aliev_panfilov.u, cmap='viridis', origin='lower')
plt.plot(sw_core['x'], sw_core['y'], 'r')
plt.title('Spiral Wave Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, n)
plt.ylim(0, n)

plt.show()
