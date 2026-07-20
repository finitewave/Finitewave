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

# set up pre-pacing protocol to stabilize the model before the main simulation:
stim_prepacing = fw.StimSingleCell(dt=0.01)
stim_prepacing.add_stim(n_beats=30, cycle_length=50., curr_value=2., duration=0.1)
stim_prepacing.add_stim(n_beats=10, cycle_length=30., curr_value=2., duration=0.1)
# create model object and set up parameters
cardiac_model = fw.AlievPanfilov()
cardiac_model.prepacing(stim_prepacing, history=True)

# create a tissue grid of size 300x300:
n = 300
tissue = fw.CardiacTissue([n, n], dr=0.25)

# set up stimulation parameters:
stim_sequence = fw.StimS1S2Cross(tissue, s1_time=0, s2_time=23, voltage_value=1)

# create model object and set up parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=200, backend="mlx")
simulation.cardiac_model = cardiac_model
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence

# run the model:
simulation.run()

# show the potential map at the end of calculations:
u = simulation.cardiac_model.output("u")
fig = plt.figure()
plt.imshow(u, cmap="coolwarm")
plt.colorbar(label="Membrane Potential")
plt.show()

fig.savefig("reentry_2d.png", dpi=300)
