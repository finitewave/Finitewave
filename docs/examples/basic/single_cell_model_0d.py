import finitewave as fw
import numpy as np
import matplotlib.pyplot as plt

cardiac_model = fw.LuoRudy91()

stim_sequence = fw.StimSingleCell(dt=0.01)
stim_sequence.add_stim(n_beats=10, cycle_length=1000, curr_value=20, duration=2)

cell_model = fw.SingleCellModel()
cell_model.cardiac_model = cardiac_model
cell_model.stim_sequence = stim_sequence
state_vars = cell_model.run(history=True)

plt.plot(cell_model.times, cell_model.u_history)
plt.xlabel("Time (ms)")
plt.ylabel("Transmembrane Potential (mV)")
plt.title(f"Single Cell Simulation: {cardiac_model.__class__.__name__}")
plt.show()
