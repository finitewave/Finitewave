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
n = 200

tissue = fw.CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.0015
aliev_panfilov.dr = 0.1
aliev_panfilov.t_max = 50

# induce the spiral wave:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1,
                                             0, n,
                                             0, 5))

tracker_sequence = fw.TrackerSequence()
# create an ECG tracker:
ecg_tracker = fw.ECG2DTracker()
ecg_tracker.start_time = 0
ecg_tracker.step = 100
ecg_tracker.measure_coords = np.array([[n//2, n//2, 10],
                                       [n//4, n//2, 10],
                                       [3*n//4, 3*n//4, 10]])

tracker_sequence.add_tracker(ecg_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

colors = ['tab:blue', 'tab:orange', 'tab:green']
plt.figure()
for i, y in enumerate(ecg_tracker.output.T):
    x = np.arange(len(y)) * aliev_panfilov.dt * ecg_tracker.step
    plt.plot(x, y, '-o', color=colors[i], label='precomputed distances')

plt.legend(title='ECG computed with')
plt.show()
