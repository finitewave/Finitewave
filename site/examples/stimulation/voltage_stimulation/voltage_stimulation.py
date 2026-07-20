"""
Using StimVoltageCoord in 2D Tissue
=====================================

Overview:
---------
This example demonstrates how to use the `StimVoltageCoord2D` class in Finitewave 
to apply a voltage-based stimulus to a rectangular region in a 2D cardiac tissue.

Stimulation Setup:
------------------
- The `StimVoltageCoord2D` class is used to define the stimulated region by its coordinates.
- A square region (6×6 nodes) at the center of the tissue is stimulated at t = 0 .
- The voltage value is set to 1.0, which for the Aliev-Panfilov model corresponds 
  to the peak excitation potential (resting = 0, peak = 1).

Simulation Parameters:
----------------------
- Model: Aliev-Panfilov 2D
- Grid size: 200 × 200
- Time step (dt): 0.01
- Space step (dr): 0.25
- Total simulation time: 10 

Application:
------------
This example is useful for learning how to define spatially localized voltage 
stimuli in 2D using coordinate-based methods. The `StimVoltageCoord2D` class 
is particularly useful for applying custom rectangular stimulation zones.
"""


import matplotlib.pyplot as plt

import finitewave as fw

# set up the tissue:
n = 200
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
# stimulate the center of the tissue with a square pulse (6 nodes on the side)
# we use voltage value = 1.0 V at t=0.
# The voltage value should be set between the resting potential and the peak potential of
# the model to ensure the stimulation is effective.
# in case of Aliev-Panfilov model, the resting potential is 0 and the peak potential is 1:
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=0, 
                                             volt_value=1.0, 
                                             x1=n//2 - 3, x2=n//2 + 3, 
                                             y1=n//2 - 3, y2=n//2 + 3))

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