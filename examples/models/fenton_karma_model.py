import matplotlib.pyplot as plt
import numpy as np
import finitewave as fw

# create a tissue of size 400x400 with cardiomycytes:
n = 100
m = 100
tissue = fw.CardiacTissueGrid([n, m])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(time=0, volt_value=1,
                                           x_min=0, x_max=5,
                                           y_min=0, y_max=m))

action_pot_tracker = fw.ActionPotentialGridTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[50, 3]]
action_pot_tracker.step = 1

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(action_pot_tracker)

# create model object and set up parameters:
simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.dr = 0.25
simulation.t_max = 20
simulation.cardiac_model = fw.FentonKarma()
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence
# simulation.tracker_sequence = tracker_sequence

# run the model:
simulation.run(num_of_threads=6)

plt.figure()
plt.imshow(simulation.cardiac_model.u, cmap='inferno')
plt.colorbar(label='Transmembrane Potential (u)')
plt.show()

# # plot the action potential
# plt.figure()
# time = np.arange(len(action_pot_tracker.output)) * simulation.dt
# plt.plot(time, action_pot_tracker.output, label="cell_50_3")
# plt.legend(title='Fenton-Karma')
# plt.xlabel('Time (ms)')
# plt.title('Action Potential')
# plt.grid()
# plt.show()