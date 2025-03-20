"""
Spiral Wave Formation in 2D Cardiac Tissue
==========================================

Overview:
---------
This example demonstrates how to initiate and observe a spiral wave 
in a two-dimensional cardiac tissue model using the Aliev-Panfilov equations. 
Spiral waves are a key phenomenon in cardiac electrophysiology, often linked to 
arrhythmias and reentrant activity.

Simulation Setup:
-----------------
- Tissue Grid: A 256×256 cardiac tissue domain.
- Spiral Wave Initiation:
  - First stimulus: Applied along the top boundary at time 0.
  - Second stimulus: Applied to the right half of the domain at time 50.
- Time and Space Resolution:
  - Temporal step (dt): 0.01 
  - Spatial resolution (dr): 0.3 
  - Total simulation time (t_max): 150 

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Apply two sequential stimulations:
   - The first stimulus excites a wavefront across the tissue.
   - The second stimulus, applied after a delay, breaks the wavefront, 
     leading to spiral wave formation.
3. Initialize and configure the Aliev-Panfilov model.
4. Run the simulation to observe spiral wave dynamics.
5. Visualize the final membrane potential distribution.

Spiral Wave Mechanism:
----------------------
Spiral waves emerge due to the interaction of an initial wave and a secondary 
stimulus applied at a critical time and location. These waves are relevant 
to studying:
- Reentrant arrhythmias (such as ventricular tachycardia).
- Excitation wave turbulence in cardiac tissue.
- Wavefront stability and self-sustained oscillations.

Visualization:
--------------
The final membrane potential distribution is displayed using matplotlib, 
revealing the characteristic spiral pattern.
"""


import matplotlib.pyplot as plt

import finitewave as fw

# set up the tissue:
n = 256
tissue = fw.CardiacTissue2D([n, n])


# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=0, volt_value=1,
                                             x1=0, x2=n, y1=0, y2=5))
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=50, volt_value=1,
                                             x1=n//2, x2=n, y1=0, y2=n))

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.3
aliev_panfilov.t_max = 150
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# show the potential map at the end of calculations:
plt.imshow(aliev_panfilov.u)
plt.show()
