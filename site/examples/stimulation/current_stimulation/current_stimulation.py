"""
Using StimCurrentCoord2D for Current-Based Stimulation
=======================================================

Overview:
---------
This example demonstrates how to apply a current-based stimulus in a 2D cardiac tissue 
using the `StimCurrentCoord2D` class from Finitewave.

Stimulation Setup:
------------------
- The center of the tissue is stimulated with a small square pulse (2×2 nodes).
- A current of 18 units is applied for 0.4  at t = 0.
- Unlike voltage stimulation, current-based stimulation allows effective excitation 
  of very small regions, which is especially useful for avoiding sink-source mismatch 
  problems in tightly localized areas.

Simulation Parameters:
----------------------
- Model: Aliev-Panfilov 2D
- Grid size: 200 × 200
- Time step (dt): 0.01 
- Space step (dr): 0.25
- Total simulation time: 10 

Application:
------------
This example is ideal for understanding how to trigger depolarization waves using 
current injection. The `StimCurrentCoord2D` class allows flexible control of both 
current strength and duration, enabling fine-tuned stimulus delivery.
"""

import matplotlib.pyplot as plt

import finitewave as fw

# set up the tissue:
n = 200
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
# All stimulation object have two types of stimulation: by current and by voltage.
# In case of current stimulation, we use two parameters: current strength and duration.
# stimulate the center of the tissue with a square pulse (2 nodes on the side)
# сurrent stimulation is set by current strength (18) and stimulation duration (0.4 model time units). 
# Current stimulation allows to bypass the problem of sink-source mismatch
# and stimulate even small areas of tissue to start a depolarization wave:
stim_sequence.add_stim(fw.StimCurrentCoord2D(time=0,
                                             curr_value=18,
                                             duration=0.4, 
                                             x1=n//2 - 1, x2=n//2 + 1, 
                                             y1=n//2 - 1, y2=n//2 + 1))

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 10
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

# run the model:
aliev_panfilov.run()

# show the potential map at the end of calculations:
plt.figure()
plt.imshow(aliev_panfilov.u)
plt.colorbar()
plt.show()