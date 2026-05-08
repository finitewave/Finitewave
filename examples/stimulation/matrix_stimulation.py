"""
Matrix-Based Stimulation with StimVoltageMatrix2D
=========================================================

Overview:
---------
This example demonstrates how to define complex spatial stimulation patterns 
in 2D cardiac tissue using the `StimVoltageMatrix2D` class in Finitewave.

Stimulation Setup:
------------------
- A 2D boolean matrix of the same size as the tissue is used to define the 
  stimulated regions.
- Two 10×10 square regions in opposite corners of the tissue are stimulated 
  at t = 0  with 1.0 V voltage.
- This flexible approach allows arbitrary spatial patterns and can be 
  generated from images, data arrays, or procedural logic.

Simulation Parameters:
----------------------
- Model: Aliev-Panfilov 2D
- Grid size: 200 × 200
- Time step (dt): 0.01 
- Space step (dr): 0.25
- Total simulation time: 15 

Application:
------------
This technique is ideal for:
- Designing realistic stimulation setups.
- Applying stimuli based on experimental data or anatomical maps.
- Studying the effect of spatial heterogeneity in excitation.

"""

import numpy as np
import matplotlib.pyplot as plt

import finitewave as fw

# set up the tissue:
n = 200
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
# stimulate two opposite corners of the tissue with a square pulse (10 nodes on the side)
# of 1.0 V at t=0.
# we create a 2D boolean matrix of the same size as the tissue 
# and set the stimulated nodes to True:
stimulation_area = np.full([n, n], False, dtype=bool)
stimulation_area[0:10, 0:10] = True
stimulation_area[n-10:n, n-10:n] = True

stim_sequence.add_stim(fw.StimVoltageMatrix2D(time=0, 
                                             volt_value=1.0, 
                                             matrix=stimulation_area))

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 15
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