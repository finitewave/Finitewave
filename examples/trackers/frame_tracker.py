
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import finitewave as fw
import numpy as np

# create a tissue of size 400x400 with cardiomycytes:
n = 200
tissue = fw.CardiacTissueGrid([n, n], dr=0.25)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(time=0, volt_value=1,
                                           x_min=n//2 - 3, x_max=n//2 + 3,
                                           y_min=n//2 - 3, y_max=n//2 + 3))
stim_sequence.add_stim(fw.StimVoltageCoord(time=35, volt_value=1,
                                           x_min=n//2 - 3, x_max=n//2 + 3,
                                           y_min=0, y_max=n))

frame_tracker = fw.FrameTracker(aggregate=True,
                                var_name="u",
                                output_dtype=np.float32,
                                step=100,
                                keep_shape=True)
tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(frame_tracker)

# create model object and set up parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=150, backend="jax")
simulation.cardiac_model = fw.AlievPanfilov()
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence
# run the model:
simulation.run()

data = frame_tracker.output[1:]

fig, ax = plt.subplots()
ax.set_title("Frame Tracker Output")
im = ax.imshow(data[0], cmap='coolwarm', animated=True)
text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')

def update(frame):
    im.set_array(data[frame])  # update image data
    text.set_text(f'Time: {frame_tracker.tracking_times[1+frame]:.2f} ms')  # update time text
    return (im, text)  # must return iterable

ani = FuncAnimation(
    fig,
    update,
    frames=data.shape[0],
    interval=100,   # ms between frames
    blit=False      # faster rendering
)

plt.show()
