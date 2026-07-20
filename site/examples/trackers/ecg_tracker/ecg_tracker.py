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

import finitewave as fw

# set up the tissue:
n, m = 300, 20

# induce the spiral wave:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(10, 1, 0, 5, 0, m))

tracker_sequence = fw.TrackerSequence()
# create an ECG tracker:
ecg_tracker = fw.ECGTracker(step=10)
ecg_tracker.measure_coords = np.array([[n//2, m//2, 5],
                                       [5, m//2, 5],
                                       [n//2, 0, 5],])

tracker_sequence.add_tracker(ecg_tracker)

simulation = fw.CardiacSimulation(dt=0.01, t_max=50, backend="jax")
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = fw.CardiacTissue([n, m], dr=0.1)
simulation.cardiac_model = fw.LuoRudy91()
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

simulation.run()

colors = ['tab:blue', 'tab:orange', 'tab:green']

fig, axs = plt.subplots(ncols=2, width_ratios=[0.3, 1])
axs[0].imshow(simulation.cardiac_model.output("u"), origin='lower')
for i, y in enumerate(ecg_tracker.output.T):
    coord = ecg_tracker.measure_coords[i]
    axs[0].scatter(coord[1], coord[0], color=colors[i])
    x = ecg_tracker.tracking_times
    axs[1].plot(x, y, '-', color=colors[i], label=f'{coord}')
    # add vertical lines for stimulus times
    for stim in stim_sequence.sequence:
        axs[1].axvline(stim.t, color='gray', linestyle='--', alpha=0.5)
      
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Voltage (mV)')
    axs[1].set_title('ECG Signals')

axs[1].legend(title='Electrodes')
plt.show()