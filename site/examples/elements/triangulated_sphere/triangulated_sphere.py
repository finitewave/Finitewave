from pathlib import Path
import numpy as np
import pyvista as pv
import finitewave as fw
import matplotlib.pyplot as plt

radius = 20
coords, elems = fw.build_triangulated_sphere(radius)

stim_matrix_1 = coords[:, 2] > 0.9 * radius
stim_matrix_2 = coords[:, 1] > 0.5 * radius

# create cardiac tissue object:
tissue = fw.CardiacTissueElements(coords, elems, elem_type='Triangle')

stim_prepacing = fw.StimSingleCell(dt=0.01)
stim_prepacing.add_stim(n_beats=30, cycle_length=30, curr_value=1, duration=0.5)

model = fw.AlievPanfilov()
model.prepacing(stim_prepacing)
# set up stimulation parameters:
# stim_coords = coords[:1, :]
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageMatrix(0, 1, stim_matrix_1))
stim_sequence.add_stim(fw.StimVoltageMatrix(25, 1, stim_matrix_2))

animation_tracker = fw.AnimationTracker(aggregate=True, step=100)
tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(animation_tracker)

# create model object and set up parameters:
simulation = fw.CardiacSimulation(backend="jax")
simulation.dt = 0.01
simulation.t_max = 100
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = model
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

# run the model:
simulation.run()

# pv.global_theme.transparent_background = True
# animation_tracker.write(format="gif", window_size=(1000, 1000), show_scalar_bar=False)

grid = fw.PyVistaSurfaceGrid(coords, elems)
grid['Vm'] = simulation.cardiac_model.u
grid.plot(scalars='Vm', cmap='inferno', show_edges=False)