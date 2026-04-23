
"""
Matrix Stimulation in 2D Cardiac Tissue
=======================================

Overview:
---------
This example demonstrates how to apply matrix-based stimulation
in a two-dimensional cardiac tissue model using the Fenton-Karma 
equations. Instead of a single stimulus source, this method applies 
stimulation at multiple predefined locations across the tissue.

Simulation Setup:
-----------------
- Tissue Grid: A 400×400 cardiac tissue domain.
- Multiple Stimulus Areas: Stimulation is applied at four distinct points.
- Stimulation Shape: Each stimulus is applied over a circular area (radius = 5).
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 10

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Define four circular stimulation areas using `skimage.draw.disk`.
3. Apply the stimuli as a matrix using `StimVoltageMatrix2D`.
4. Initialize and configure the Fenton-Karma model.
5. Run the simulation to observe how multiple stimulation sites influence 
   wave propagation.
6. Visualize the final membrane potential distribution.

Application:
------------
This method is useful for simulating paced activation patterns seen 
in electrophysiology studies, where multiple sites are excited 
simultaneously. It can help analyze conduction velocity, wavefront 
interactions, and reentry formation.

Visualization:
--------------
The final membrane potential distribution is displayed using matplotlib, 
showing how excitation spreads from the stimulated regions.
"""


import matplotlib.pyplot as plt
from skimage import draw
import numpy as np

import finitewave as fw

# set up cardiac tissue:
n = 400
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_area = np.full([400, 400], False, dtype=bool)
ii, jj = draw.disk([100, 100], 5)
stim_area[ii, jj] = True
ii, jj = draw.disk([100, 300], 5)
stim_area[ii, jj] = True
ii, jj = draw.disk([300, 100], 5)
stim_area[ii, jj] = True
ii, jj = draw.disk([300, 300], 5)
stim_area[ii, jj] = True
stim_sequence.add_stim(fw.StimVoltageMatrix2D(0, 1, stim_area))

# create model object:
fenton_karma = fw.FentonKarma2D()
# set up numerical parameters:
fenton_karma.dt = 0.01
fenton_karma.dr = 0.25
fenton_karma.t_max = 10
# add the tissue and the stim parameters to the model object:
fenton_karma.cardiac_tissue = tissue
fenton_karma.stim_sequence = stim_sequence

fenton_karma.run()

# show the potential map at the end of calculations:
# plt.figure()
plt.imshow(fenton_karma.u)
plt.show()
