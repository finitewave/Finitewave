"""
Electrocardiogram (ECG) Tracking in 2D Cardiac Tissue
=====================================================

Overview:
---------
This example demonstrates how to use the ECG2DTracker to record an 
electrocardiogram (ECG) from a 2D cardiac tissue simulation. The ECG 
signal is obtained from multiple measurement points at a given distance 
from the tissue.

Simulation Setup:
-----------------
- Tissue Grid: A 400×400 cardiac tissue domain.
- Stimulation:
  - A left-side stimulus is applied at time t = 0.
  - The excitation wave propagates across the tissue.
- ECG Tracking:
  - Three measurement points are positioned at increasing vertical distances.
  - The signal strength is computed using an inverse distance power law.
  - Measurement points:  
    - (n/2, n/4, 10)
    - (n/2, n/2, 10)
    - (n/2, 3n/4, 10)
  - Sampling step: Every 10 time steps.
- Time and Space Resolution:
  - Temporal step (dt): 0.001
  - Spatial resolution (dr): 0.1
  - Total simulation time (t_max): 50

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Apply stimulation along the left boundary.
3. Set up an ECG tracker:
   - Records electrical activity from multiple measurement points.
   - Uses an inverse distance weighting (power = 2) to compute the 
     potential at each location.
4. Run the Aliev-Panfilov model to simulate cardiac wave propagation.
5. Extract and visualize the ECG waveform.

Application:
------------
ECG tracking in a simulated tissue is useful for:
- Studying ECG signal characteristics in controlled environments.
- Understanding the relationship between wave propagation and ECG morphology.
- Testing the effect of different tissue properties on the ECG signal.

Visualization:
--------------
The recorded ECG signal is plotted over time using matplotlib, 
illustrating how electrical wave activity in cardiac tissue translates 
into an observable ECG trace.

"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave.gridywave as fw

# set up the tissue:
n = 100

# 
cardiac_model = fw.Courtemanche()
cardiac_model.gkur_coeff *= 0.5
cardiac_model.gto *= 0.5
cardiac_model.gcal *= 0.3

# induce the spiral wave:
stim_sequence = fw.StimSequence()

for i in range(10):
    stim_time = i * 300
    stim_sequence.add_stim(fw.StimVoltageGridCoord(stim_time, 1,
                                                   0, n,
                                                   0, 5))

tracker_sequence = fw.TrackerSequence()
# create an ECG tracker:
ecg_tracker = fw.ECGGridTracker()
ecg_tracker.start_time = 5
ecg_tracker.step = 100
ecg_tracker.measure_coords = np.array([[n//2, n//2, 20],
                                       [10, n//2, 20],
                                       [n//2, 3*n//4, 20],])

tracker_sequence.add_tracker(ecg_tracker)

simulation = fw.CardiacGridSimulation()
simulation.dt = 0.01
simulation.dr = 0.25
simulation.t_max = 1000
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = fw.CardiacTissueGrid([n, n])
simulation.cardiac_model = cardiac_model
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

simulation.run()

colors = ['tab:blue', 'tab:orange', 'tab:green']

fig, axs = plt.subplots(ncols=2)
axs[0].imshow(simulation.diffusion_model.u)
for i, y in enumerate(ecg_tracker.output.T):
    coord = ecg_tracker.measure_coords[i]
    axs[0].scatter(coord[1], coord[0], color=colors[i])
    x = (ecg_tracker.start_time +
         np.arange(len(y)) * simulation.dt * ecg_tracker.step)
    axs[1].plot(x, y, '-o', color=colors[i], label=f'{coord}')

axs[1].legend(title='Electrodes')
plt.show()