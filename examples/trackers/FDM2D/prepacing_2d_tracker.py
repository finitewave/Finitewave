import matplotlib.pyplot as plt
import finitewave as fw

path = "."
stim_times = [0]
pacing_time = 1
voltage_threshold = 0
values = -20

tissue = fw.CardiacTissueFDM((400, 50))

for stim_time in stim_times:
    stim = fw.StimVoltageCoordFDM(stim_time, values, 0, 5, 0, 50)

stim_sequence = fw.StimSequence()
stim_sequence.add_stim(stim)

prepacing_tracker = fw.PrepacingTrackerFDM()
prepacing_tracker.path = path
prepacing_tracker.cell_ind = [50, 5]
prepacing_tracker.pacing_time = pacing_time
prepacing_tracker.voltage_threshold = voltage_threshold

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(prepacing_tracker)

model = fw.CourtemancheFDM()
model.cardiac_tissue = tissue
model.dt = 0.01
model.dr = 0.25
model.t_max = 200
model.stim_sequence = stim_sequence
model.tracker_sequence = tracker_sequence

model.run()

plt.imshow(model.u)
plt.show()


# prepacing_tracker.write()