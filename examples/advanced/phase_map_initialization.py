
import matplotlib.pyplot as plt

import finitewave as fw
import numpy as np


def build_spiral_core(mesh, levels=100, phase_offset=0, clockwise=True):
    """
    Builds a core of a spiral wave by assigning phase values to each point in the mesh
    based on its angle from the center.
    
    Parameters
    ----------
    mesh : numpy.ndarray
        A 2D array representing the tissue grid.
    levels : int, optional
        The number of phase levels to use (default is 100).
    phase_offset : float, optional
        The phase offset for the spiral wave (default is 0).
    clockwise : bool, optional
        Whether the spiral wave rotates clockwise (default is True).

    Returns
    -------
    numpy.ndarray
        A 2D array of phase values for each point in the mesh.
    """
    n, m = mesh.shape
    x = np.arange(n) - n // 2
    y = np.arange(m) - m // 2
    X, Y = np.meshgrid(x, y, indexing='ij')
    theta = np.arctan2(Y, X)
    theta += phase_offset
    theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi  # Wrap to [-pi, pi]
    if not clockwise:
        theta = -theta
    theta = np.digitize(theta, np.linspace(-np.pi, np.pi, levels)) - 1
    return theta.astype(int)


# create a tissue of size 50x10 to collect the state variables.
n, m = 50, 5
tissue = fw.CardiacTissue([n, m], dr=0.25)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimCurrentCoord(time=0, curr_value=10, duration=0.1,
                                           x_min=0, x_max=1, y_min=0, y_max=m))

stim_prepacing = fw.StimSingleCell(dt=0.01)
stim_prepacing.add_stim(n_beats=5, cycle_length=40, curr_value=1, duration=0.1)
stim_prepacing.add_stim(n_beats=5, cycle_length=30, curr_value=1, duration=0.1)
stim_prepacing.add_stim(n_beats=5, cycle_length=25, curr_value=1, duration=0.1)

model = fw.AlievPanfilov()
model.prepacing(stim_prepacing)

state_tracker = fw.MultiVariableTracker(node_inds=[(n//2, m//2)], step=10, start_time=5)
tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(state_tracker)

# create model object and set up parameters:
simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.t_max = 30
simulation.cardiac_model = model
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence
simulation.tracker_sequence = tracker_sequence

# run the model:
simulation.run()

# create a tissue of size 400x400 and place 4 spirals in it.
n = 400

spiral_map = np.zeros((n, n), dtype=int)
for i in range(2):
    for j in range(2):
        x_min, x_max = i * n//2, (i+1) * n//2
        y_min, y_max = j * n//2, (j+1) * n//2
        sub_mesh = spiral_map[x_min:x_max, y_min:y_max]
        sub_map = build_spiral_core(sub_mesh, levels=len(state_tracker.tracking_times),
                                    phase_offset=i*np.pi, clockwise=(i+j)%2==0)
        spiral_map[x_min:x_max, y_min:y_max] = sub_map


tissue = fw.CardiacTissue([n, n], dr=0.25)

model = fw.AlievPanfilov()
model.prepacing(stim_prepacing)

# create model object and set up parameters:
simulation = fw.CardiacSimulation(backend='mlx')
simulation.dt = 0.01
simulation.t_max = 100
simulation.cardiac_model = model
simulation.cardiac_tissue = tissue

simulation.initialize()

u = state_tracker.output['u'][spiral_map]
v = state_tracker.output['v'][spiral_map]

state_vars = {'u': u.flatten(),
              'v': v.flatten()}

model.update_state_variables(state_vars)

# run the model:
simulation.run(initialize=False)

u = model.output("u")

# show the potential map at the end of calculations:
fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
axs[0].imshow(spiral_map, cmap='viridis', origin='lower')
axs[0].set_title('Initial Phase Map')
axs[1].imshow(u, cmap='inferno', origin='lower')
axs[1].set_title('Transmembrane Potential')
plt.show()
