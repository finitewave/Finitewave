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
import numpy as np
import mlx.core as mx


stim_prepacing = fw.StimPrepacing(dt=0.005)
stim_prepacing.add_stim(n_beats=30, cycle_length=1000., curr_value=20., duration=2.)
stim_prepacing.add_stim(n_beats=30, cycle_length=500., curr_value=20., duration=2.)

# create model object and set up parameters
courtemanche = fw.TenTusscherPanfilov2006()
courtemanche.prepacing(stim_prepacing)


# create a tissue of size 400x400 with cardiomycytes:
n = 400
tissue = fw.CardiacTissueGrid([n, n], dr=0.25)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(time=0, volt_value=1,
                                           x_min=0, x_max=n//2,
                                           y_min=0, y_max=n))
stim_sequence.add_stim(fw.StimVoltageCoord(time=310, volt_value=1,
                                           x_min=0, x_max=n,
                                           y_min=0, y_max=n//2))
# create model object and set up parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=500)
simulation.cardiac_model = courtemanche
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence

# run the model:
simulation.run()

u = simulation.cardiac_model.output("u")

# show the potential map at the end of calculations:
plt.figure()
plt.imshow(u, cmap="inferno")
plt.colorbar(label="Membrane Potential")
plt.show()
