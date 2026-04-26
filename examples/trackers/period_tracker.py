

import finitewave as fw

# number of nodes on the side
n = 200
tissue = fw.CardiacTissueGrid([n, n], dr=0.3)

model = fw.AlievPanfilov()

# induce spiral wave:
stim_sequence = fw.StimS1S2Cross(tissue, s1_time=0, s2_time=31, voltage_value=1)

# set up the tracker:
period_tracker = fw.PeriodTracker(node_inds=[[5, 5], [n//2, n//2]],
                                  threshold=0.5, step=10, start_time=100)

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(period_tracker)

# set up the simulation:
simulation = fw.CardiacSimulation(dt=0.01, t_max=300, backend="mlx")
simulation.cardiac_tissue = tissue
simulation.cardiac_model = model
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

# run the simulation:
simulation.run()

print("Periods at tracked nodes:")
for i, period in enumerate(period_tracker.output):
    # print with 2 decimal places
    print(f"Node {i}: {period.round(2)}")