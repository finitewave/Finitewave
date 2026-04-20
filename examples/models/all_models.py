"""
Running the Aliev-Panfilov Model in 2D
======================================

Overview:
---------
This example demonstrates how to run a basic 2D simulation of the 
Aliev-Panfilov model using the Finitewave framework. 

Simulation Setup:
-----------------
- Tissue Grid: A 100×5 cardiac tissue domain.
- Stimulation:
  - A square side stimulus is applied at t = 0.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 50

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Apply a stimulus along the upper boundary to initiate excitation.
3. Set up and run the Aliev-Panfilov model.
4. Visualize the transmembrane potential.

"""

import matplotlib.pyplot as plt
import numpy as np
import finitewave as fw


models = [{"model": fw.Barkley(), "t_max": 6, "title": 'Barkley'},
          {"model": fw.AlievPanfilov(), "t_max": 60, "title": 'Aliev-Panfilov'},
          {"model": fw.MitchellSchaeffer(), "t_max": 500, "title": 'Mitchell-Schaeffer'},
          {"model": fw.FentonKarma(), "t_max": 500, "title": 'Fenton-Karma'},
          {"model": fw.BuenoOrovio(), "t_max": 500, "title": 'Bueno-Orovio2D'},
          {"model": fw.LuoRudy91(), "t_max": 500, "title": 'Luo-Rudy91'},
          {"model": fw.Courtemanche(), "t_max": 500, "title": 'Courtemanche'},
          {"model": fw.TenTusscherPanfilov2006(), "t_max": 500, "title": 'TP06'}]


act_pot = []
for model_dict in models:
    model = model_dict["model"]
    t_max = model_dict["t_max"]
    title = model_dict["title"]
    # create a tissue of size 100x10 with cardiomycytes:
    n = 100
    m = 10
    tissue = fw.CardiacTissueGrid([n, m], dr=0.25)

    # set up stimulation parameters:
    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(fw.StimVoltageCoord(time=5, volt_value=1,
                                               x_min=0, x_max=5,
                                               y_min=0, y_max=m))

    action_pot_tracker = fw.ActionPotentialTracker()
    action_pot_tracker.node_inds = [[n//2, m//2]]
    action_pot_tracker.step = 1

    tracker_sequence = fw.TrackerSequence()
    tracker_sequence.add_tracker(action_pot_tracker)

    # create model object and set up parameters:
    simulation = fw.CardiacSimulation(dt=0.01, t_max=t_max)
    simulation.cardiac_model = model
    simulation.cardiac_tissue = tissue
    simulation.stim_sequence = stim_sequence
    simulation.tracker_sequence = tracker_sequence

    # run the model:
    simulation.run()

    act_pot.append(action_pot_tracker.output)

# plot the action potential
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
for i, model_dict in enumerate(models):
    title = model_dict["title"]
    time = np.arange(len(act_pot[i])) * simulation.dt
    axs[i//2, i%2].plot(time, act_pot[i], label=f"cell_{n//2}_{m//2}")
    axs[i//2, i%2].set_xlabel('Time (ms)')
    axs[i//2, i%2].set_ylabel('Voltage (mV)')
    axs[i//2, i%2].set_title(title)
    axs[i//2, i%2].grid()
plt.tight_layout()
plt.show()
