"""
Prepacing a Single Cell Model
==============================

Overview:
---------
This example demonstrates how to perform prepacing of a single cell model
using the Finitewave framework. Prepacing is a technique used to bring
a cardiac cell model to a steady state before running a full simulation.

Simulation Setup:
-----------------
- Time step (dt): 0.01 ms
- Prepacing Sequence:
  - 3 beats with a cycle length of 1000 ms and a stimulus duration of 2 ms.
  - 2 beats with a cycle length of 500 ms and a stimulus duration of 2 ms.
  - 2 beats with a cycle length of 300 ms and a stimulus duration of 2 ms.
- Model: Courtemanche single cell model.

Execution:
----------
1. Create a `StimSingleCell` instance and define the prepacing sequence.
2. Initialize the `Courtemanche` model and run the prepacing.
3. Plot the transmembrane potential and stimulus current over time.

"""

import matplotlib.pyplot as plt
import finitewave as fw

prepacing = fw.StimSingleCell(dt=0.01)
prepacing.add_stim(n_beats=3, cycle_length=1000, curr_value=20., duration=2.)
prepacing.add_stim(n_beats=2, cycle_length=500, curr_value=20., duration=2.)
prepacing.add_stim(n_beats=2, cycle_length=300, curr_value=20., duration=2.)

model = fw.Courtemanche()
model.prepacing(prepacing, history=True)

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=False, height_ratios=[3, 1])
axs[0].plot(model.pacing_times, model.u_pacing, label='V')
axs[0].set_ylabel('V (mV)')
axs[0].set_title('Single Cell Prepacing')
axs[1].plot(model.pacing_times, model.pacing_stims, label='Vstim', color='tab:orange')
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('I (pA)')
axs[1].set_title('Stimulus Current')
plt.tight_layout()
plt.show()
