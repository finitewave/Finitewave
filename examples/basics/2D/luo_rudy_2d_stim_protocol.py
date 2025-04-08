"""
S1-S3 Stimulation Protocol in 2D Luo-Rudy Model
===============================================

Overview:
---------
This example demonstrates how to apply the S1-S3 stimulation protocol 
in a two-dimensional Luo-Rudy 1991 (LR91) cardiac tissue model. The 
protocol involves multiple stimulations delivered sequentially from a 
point electrode, allowing the study of wave propagation and refractory 
periods in excitable tissue.

Simulation Setup:
-----------------
- Tissue Grid: A 256×256 cardiac tissue domain.
- Luo-Rudy 1991 Model: Simulates cardiac action potential dynamics.
- Stimulation:
  - S1-S3 protocol: Three stimulus pulses are applied at different times (0 ms, 150 ms, and 290 ms).
  - Stimulus source: A point electrode with a radius of 5 units.
  - Stimulus current: Applied for 30 ms at each step.
- Time and Space Resolution:
  - Temporal step (dt): 0.01 ms
  - Spatial resolution (dr): 0.3 mm
  - Total simulation time (t_max): 700 ms

Execution:
----------
1. Create a 2D cardiac tissue grid with heterogeneous conductivity using a 
   diffusion pattern (`Diffuse2DPattern`).
2. Apply three sequential stimulations at different time intervals.
3. Set up and initialize the Luo-Rudy model.
4. Run the simulation using multithreading (16 threads) for performance.
5. Visualize the final membrane potential distribution.

S1-S3 Stimulation Protocol:
---------------------------
- The S1 stimulus initiates normal excitation propagation.
- The S2 stimulus is delivered after a short delay to test refractoriness.
- The S3 stimulus assesses the response of the tissue to multiple stimulations, 
  potentially triggering reentry or arrhythmias.

Visualization:
--------------
The final membrane potential distribution is displayed using matplotlib, 
providing insight into excitation wavefront interactions under repeated 
stimulation.
"""

import matplotlib.pyplot as plt
import finitewave as fw

n = 256
# create mesh
tissue = fw.CardiacTissue2D((n, n))
tissue.add_pattern(fw.Diffuse2DPattern(0.3))

# set up stimulation parameters
stim_sequence = fw.StimSequence()
for t in [0, 400, 370]:
    stim = fw.StimCurrentArea2D(t, 30, 1)
    stim.add_stim_point([5, n//2], tissue.mesh, size=5)
    stim_sequence.add_stim(stim)


# create model object and set up parameters
luo_rudy = fw.LuoRudy912D()
luo_rudy.dt = 0.01
luo_rudy.dr = 0.3
luo_rudy.t_max = 160

# add the tissue and the stim parameters to the model object
luo_rudy.cardiac_tissue = tissue
luo_rudy.stim_sequence = stim_sequence

# run the model
luo_rudy.run(num_of_theads=16) # thReads - fix

plt.imshow(luo_rudy.u)
plt.colorbar()
plt.show()
