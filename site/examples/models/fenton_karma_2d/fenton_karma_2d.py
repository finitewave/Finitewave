import matplotlib.pyplot as plt
import numpy as np
import finitewave as fw


stim_prepacing = fw.StimSingleCell(dt=0.01)
stim_prepacing.add_stim(n_beats=30, cycle_length=1000., curr_value=1., duration=.2)

# create model object and set up parameters
fenton_karma = fw.FentonKarma()
fenton_karma.prepacing(stim_prepacing)

# create a tissue of size 400x400 with cardiomycytes:
n = 100
m = 10
tissue = fw.CardiacTissue([n, m], dr=0.25)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(time=0, volt_value=1,
                                           x_min=0, x_max=5,
                                           y_min=0, y_max=m))

action_pot_tracker = fw.ActionPotentialTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.node_inds = [[n//2, m//2]]
action_pot_tracker.step = 1

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(action_pot_tracker)

# create model object and set up parameters:
simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.t_max = 500
simulation.cardiac_model = fenton_karma
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

# run the model:
simulation.run(num_of_threads=6)

# plot the action potential
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))

time = np.arange(len(fenton_karma.u_pacing)) * simulation.dt
axs[0].plot(time, fenton_karma.u_pacing, label="cell_50_3")
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Voltage (mV)')
axs[0].set_title('Prepacing Protocol')
axs[0].grid()

time = np.arange(len(action_pot_tracker.output)) * simulation.dt
axs[1].plot(time, action_pot_tracker.output, label="cell_50_3")
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Voltage (mV)')
axs[1].set_title('Action Potential')
axs[1].grid()
plt.tight_layout()
plt.show()