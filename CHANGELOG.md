# Changelog

## [0.9.3] May 2026

### Added

- Initial conditions support. Model variables can now be initialized either as scalars (uniform across the tissue)
  or as arrays (node-wise values). All initial condition fields use the `init_*` prefix.
  See the **change_initial_conditions.py** example.

- Courant–Friedrichs–Lewy (CFL) condition check to warn about potential instability of the user-defined
  numerical parameters.

- Propagation tests for validating model conduction velocity.

- **Reentry and Spiral waves** tutorial.

### Fixes

- Adjusted diffusion coefficients (`D_model`) in Fenton–Karma and Mitchell–Schaeffer models
  to ensure physiological conduction velocities.

- Removed the unnecessary variable `irel` in the Courtemanche model (now treated as a parameter).

- Standardized variable and parameter names in the ten-Tusscher–Panfilov 2006 model.

- Fixed an API issue and a potential deadlock in AnimationBuilder2D/3D.


## [0.9.0] March 2026

### Added

- Node-specific model parameters. You can now define model parameters as arrays with individual values for each mesh node. See the **parameter_regions.py** example.

- External model library support. Cardiac models are now maintained in separate repositories and used as dependencies by Finitewave.

- Observers. You can now attach observers that are evaluated during the simulation. See the **observers.py** example.

- Lazy imports and environment checks for heavy or 'problematic' dependencies.

---

### Changed

- Removed **2D/3D** prefixes from class names.  
  Example: use **CardiacTissue** instead of **CardiacTissue2D** or **CardiacTissue3D**.  
  The framework automatically adjusts dimensionality based on the mesh.

- Kernel generation system. Finitewave now uses kernel generators to compose computational steps.  
  This enables node-specific parameters and observer integration.

- **VariablesTracker** replaces both **VariableTracker** and **MultiVariableTracker**.  
  Use **VariablesTracker** in the same way as the former **MultiVariableTracker**.

- Finitewave is now available via **pip**.

---

### Notes

- The old **2D/3D** class prefixes are still supported for backward compatibility.  
  Existing scripts remain compatible with this version.
