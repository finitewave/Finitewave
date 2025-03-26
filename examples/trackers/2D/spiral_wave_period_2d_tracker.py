"""
Measuring Wave Period in 2D Cardiac Tissue
==========================================

Overview:
---------
This example demonstrates how to use the `Period2DTracker` to measure the 
wave period at specific locations in a 2D cardiac tissue simulation. 
This is particularly useful for analyzing repetitive wave activity, such as 
spiral waves or regular pacing, and for determining local cycle lengths.

Simulation Setup:
-----------------
- Tissue Grid: A 200×200 cardiac tissue domain.
- Stimulation:
  - First stimulus applied to the bottom half of the domain at t = 0.
  - Second stimulus applied to the left half at t = 31 to initiate reentry.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 300

Wave Period Tracking:
---------------------
- A list of detector positions is provided through the `cell_ind` parameter:
  - Positions: (1,1), (5,5), (7,3), (9,1), (100,100), (150,3), (100,150)
- The tracker monitors threshold crossings at each specified cell to calculate 
  the local activation period.
- Tracking starts at `start_time = 100` and is evaluated every `step = 10` steps.
- Threshold voltage for detection is set to `0.5`.

Execution:
----------
1. Create and configure a 2D cardiac tissue grid.
2. Apply sequential stimulation to induce spiral or repetitive wave activity.
3. Configure the `Period2DTracker` with desired cell indices.
4. Run the Aliev-Panfilov model and track the period at each specified location.
5. Plot the average and standard deviation of the measured periods.

Application:
------------
- Useful for analyzing cycle lengths during sustained wave activity.
- Applicable in reentry studies, tissue heterogeneity analysis, or pacing experiments.
- Helps evaluate the spatial variability of wave dynamics and rhythm regularity.

Output:
-------
The resulting plot shows the mean wave period at each detector location, 
along with error bars indicating standard deviation over time.

"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 200
tissue = fw.CardiacTissue2D([n, n])

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt    = 0.01
aliev_panfilov.dr    = 0.25
aliev_panfilov.t_max = 300

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, n//2))
stim_sequence.add_stim(fw.StimVoltageCoord2D(31, 1, 0, n//2, 0, n))

tracker_sequence = fw.TrackerSequence()
# add action potential tracker
# # add period tracker:
period_tracker = fw.Period2DTracker()
# Here we create an int array of detectors as a list of positions in which we want to calculate the period.
positions = np.array([[1, 1], [5, 5], [7, 3], [9, 1],
                      [100, 100], [150, 3], [100, 150]])
period_tracker.cell_ind = positions
period_tracker.threshold = 0.5
period_tracker.start_time = 100
period_tracker.step = 10
tracker_sequence.add_tracker(period_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue   = tissue
aliev_panfilov.stim_sequence    = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

# get the wave period as a pandas Series with the cell index as the index:
periods = period_tracker.output

# plot the wave period:
plt.figure()
plt.errorbar(range(len(positions)),
             periods.apply(lambda x: x.mean()),
             yerr=periods.apply(lambda x: x.std()),
             fmt='o')
plt.xticks(range(len(positions)), [f'({x[0]}, {x[1]})' for x in positions],
           rotation=45)
plt.xlabel('Cell Index')
plt.ylabel('Period')
plt.title('Wave period')
plt.tight_layout()
plt.show()