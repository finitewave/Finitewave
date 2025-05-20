"""
StateKeeper Example: Saving and Loading Simulation States
=========================================================

Overview:
---------
This example demonstrates how to use the StateKeeper functionality in 
Finitewave to save and restore the state of a 2D cardiac simulation. 
This allows a simulation to be paused and resumed at a later time 
without restarting from the beginning.

Key Features:
-------------
- State Saving: The model saves intermediate states at specific times.
- State Loading: The simulation is resumed from a saved state.
- Multiple Runs: The model is executed in three phases:
  1. First run (0 - 20): The initial simulation run.
  2. Second run (10 - 20): Resumes from a saved state at t = 10.
  3. Third run (20 - 30): Resumes from a saved state at t = 20.

Simulation Setup:
-----------------
- Tissue Grid: A 400×400 cardiac tissue domain.
- Stimulation: A small localized stimulus applied at the center.
- State Saving:
  - The state is saved at t = 10 ("state_0").
  - The final state is saved at t = 20 ("state_1").
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - First run duration: 0 - 20
  - Second and third run durations: 10 - 20 each.

Execution Workflow:
-------------------
1. Run the first simulation and save the state at t = 10 and t = 20.
2. Delete the model instance and clear memory using `gc.collect()`.
3. Create a new model instance and load "state_0", then continue the 
   simulation from t = 10 to 20.
4. Create another new instance, load "state_1", and run from t = 20 to 30.
5. Visualize the results:
   - First run (t=0 to 20)
   - Second run (t=10 to 20)
   - Third run (t=20 to 30)
6. Delete saved states to clean up temporary files.

Application:
------------
State saving is useful for:
- Long simulations: Avoids restarting from scratch in case of interruptions.
- Parameter tuning: Allows resuming simulations from intermediate states.
- Multi-stage analysis: Investigates different scenarios from a common starting point.

Visualization:
--------------
The final results are displayed using matplotlib, showing the progression of 
the simulation across the three phases.
"""

import matplotlib.pyplot as plt
import gc
import shutil

import finitewave as fw

# number of nodes on the side
# create a tissue of size 400x400 with cardiomycytes:
n = 400
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, n//2 - 5, n//2 + 5,
                                             n//2 - 5, n//2 + 5))

# set up state saver parameters:
# to save only one state you can use StateSaver directly
state_savers = fw.StateSaverCollection()
state_savers.savers.append(fw.StateSaver("state_0", time=10)) # will save at t=10
state_savers.savers.append(fw.StateSaver("state_1")) # will save at t=20 (at the end of the run)

# create model object and set up parameters:
mitchell_schaeffer = fw.MitchellSchaeffer2D()
mitchell_schaeffer.dt = 0.01
mitchell_schaeffer.dr = 0.25
mitchell_schaeffer.t_max = 20
# add the tissue and the stim parameters to the model object:
mitchell_schaeffer.cardiac_tissue = tissue
mitchell_schaeffer.stim_sequence = stim_sequence
mitchell_schaeffer.state_saver = state_savers

# run the model:
mitchell_schaeffer.run()

u_before = mitchell_schaeffer.u.copy()

# We delete model and use gc.collect() to ask the virtual machine remove
# objects from memory. Though it's not necessary to do this.
del mitchell_schaeffer
gc.collect()

# # # # # # # # #

# Here we create a new model and load states from the previous calculation to
# continue.


# recreate the model
mitchell_schaeffer = fw.MitchellSchaeffer2D()

# set up numerical parameters:
mitchell_schaeffer.dt = 0.01
mitchell_schaeffer.dr = 0.25
mitchell_schaeffer.t_max = 10
# add the tissue and the state_loader to the model object:
mitchell_schaeffer.cardiac_tissue = tissue
mitchell_schaeffer.state_loader = fw.StateLoader("state_0")

mitchell_schaeffer.run()
u_after = mitchell_schaeffer.u.copy()

# recreate the model
mitchell_schaeffer = fw.MitchellSchaeffer2D()

# set up numerical parameters:
mitchell_schaeffer.dt = 0.01
mitchell_schaeffer.dr = 0.25
mitchell_schaeffer.t_max = 10
# add the tissue and the state_loader to the model object:
mitchell_schaeffer.cardiac_tissue = tissue
mitchell_schaeffer.state_loader = fw.StateLoader("state_1")

mitchell_schaeffer.run()

# plot the results
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(u_before)
axs[0].set_title("First run from t=0 to t=20")
axs[1].imshow(u_after)
axs[1].set_title("Second run from t=10 to t=20")
axs[2].imshow(mitchell_schaeffer.u)
axs[2].set_title("Third run from t=20 to t=30")
plt.show()

# remove the state directory
shutil.rmtree("state_0")
shutil.rmtree("state_1")
