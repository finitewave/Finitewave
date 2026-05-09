
"""
 Electrode Stimulation in 2D Cardiac Tissue
=======================================

Overview:
---------
This example demonstrates how to apply electrode-based stimulation
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
2. Define four point stimulation coordinates.
3. Apply the stimuli using `StimVoltageElectrodes`.
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
tissue = fw.CardiacTissue([n, n], dr=0.25)

stim_coords = [[100, 100], [100, 300], [300, 100], [300, 300]]

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageElectrodes(0, 1, stim_coords, size=5))

# create model object:
fenton_karma = fw.FentonKarma()
# set up numerical parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=10)
simulation.cardiac_model = fenton_karma
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence

simulation.run()

# show the potential map at the end of calculations:
# plt.figure()
plt.imshow(fenton_karma.output("u"), cmap="inferno")
plt.colorbar(label="Membrane Potential")
plt.title("Membrane Potential Distribution After Electrode Stimulation")
plt.show()
