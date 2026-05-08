"""
Spiral Wave Breakup and Induced Chaos (Aliev-Panfilov 2D)
==========================================================

Overview:
---------
This example demonstrates how to initiate a spiral wave in a 2D excitable 
medium using the Aliev-Panfilov model and subsequently destabilize it with 
two additional stimuli. This approach leads to spiral wave breakup and the 
onset of chaotic, fibrillation-like activity in a homogeneous tissue.

Simulation Setup:
-----------------
- Tissue Grid: A 200×200 homogeneous cardiac tissue domain.
- Model: Aliev-Panfilov 2D model.
- Stimulation Protocol:
  - **S1 (t = 0 ms)**: Planar stimulus to the top half of the tissue.
  - **S2 (t = 31 ms)**: Vertical stimulus on the left half to induce wave rotation (spiral).
  - **S3–S4 (t = 75 ms, 125 ms)**: Localized current pulses in the bottom center 
    to destabilize the spiral and trigger wave breakup.
- Time and Space Resolution:
  - Temporal step (dt): 0.01 ms
  - Spatial resolution (dr): 0.25 mm
  - Total simulation time (t_max): 195 ms

Execution:
----------
1. A planar wave is launched at the top to propagate downward.
2. A second stimulus creates a partial reentry and initiates a spiral.
3. Two well-timed localized stimuli are applied near the spiral core, 
   leading to fragmentation and chaotic wave propagation.
4. The model is integrated over time to observe the evolution of excitation.

Expected Outcome:
-----------------
- Formation of a spiral wave pattern.
- Spiral destabilization due to extra stimuli.
- Emergence of complex, self-sustaining chaotic patterns resembling electrical fibrillation.

Visualization:
--------------
The final membrane potential is visualized using matplotlib. 
Chaotic activity is indicated by irregular, fragmented wavefronts.

"""
import numpy as np
import matplotlib.pyplot as plt
import finitewave as fw

n = 200
tissue = fw.CardiacTissueGrid((n, n), dr=0.25)

stim_sequence = fw.StimSequence()

stim_sequence.add_stim(fw.StimVoltageCoord(0, 1, 0, n, 0, n//2))
stim_sequence.add_stim(fw.StimVoltageCoord(31, 1, 0, n//2, 0, n))
# extra stimuli to break the spiral waves: 
stim_sequence.add_stim(fw.StimCurrentCoord(75, 3, 3, 90, 100, n//2, n))
stim_sequence.add_stim(fw.StimCurrentCoord(125, 3, 3, 90, 100, n//2, n))

# Set up the Aliev-Panfilov model:
stim_prepacing = fw.StimSingleCell(dt=0.01)
stim_prepacing.add_stim(n_beats=30, cycle_length=30., curr_value=2., duration=.1)

aliev_panfilov = fw.AlievPanfilov()
aliev_panfilov.prepacing(stim_prepacing)

simulation = fw.CardiacSimulation(dt=0.01, t_max=195, backend="numba")
simulation.cardiac_model = aliev_panfilov
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence

simulation.run()

plt.imshow(aliev_panfilov.output(), cmap='plasma')
plt.colorbar(label="Membrane Potential")
plt.title("Chaotic pattern")
plt.show()
