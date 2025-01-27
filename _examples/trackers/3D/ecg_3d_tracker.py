
#
#

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 200
nk = 50

tissue = fw.CardiacTissue3D([n, n, nk])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n, nk], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 100

# induce the spiral wave:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, n, 0, 100, 0, nk))
stim_sequence.add_stim(fw.StimVoltageCoord3D(31, 1, 0, 100, 0, n, 0, nk))

tracker_sequence = fw.TrackerSequence()
# create an ECG tracker:
ecg_tracker = fw.ECG3DTracker()
ecg_tracker.start_time = 40
ecg_tracker.step = 10
ecg_tracker.measure_coords = np.array([[n//2, n//2, nk+3] * 1]).reshape(1, 3)
# # create an ECG tracker with memory save:
# ecg_tracker_memsave = fw.ECG3DTracker(memory_save=True)
# ecg_tracker_memsave.start_time = 40
# ecg_tracker_memsave.step = 10
# ecg_tracker_memsave.measure_coords = np.array([[n//2, n//2, nk+3],
#                                                [n//4, n//2, nk+3],
#                                                [3*n//4, 3*n//4, nk+3]])

tracker_sequence.add_tracker(ecg_tracker)
# tracker_sequence.add_tracker(ecg_tracker_memsave)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()


# colors = ['tab:blue', 'tab:orange', 'tab:green']
# plt.figure()
# for i, y in enumerate(ecg_tracker.output.T):
#     x = np.arange(len(y)) * aliev_panfilov.dt * ecg_tracker.step
#     plt.plot(x, y, color=colors[i], label='precomputed distances')

# for i, y in enumerate(ecg_tracker_memsave.output.T):
#     plt.plot(x, y, 'o', color=colors[i], label='memory save')

# plt.legend(title='ECG computed with')
# plt.savefig("ecg_3d_tracker.png")
# plt.close()
