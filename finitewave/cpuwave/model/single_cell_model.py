
import numpy as np
from finitewave.cpuwave.model.kernel.single_cell_numba_kernel import (
    SingleCellNumbaKernel,
)


class SingleCellModel:
    def __init__(self):
        self.kernel_generator = SingleCellNumbaKernel()

    def initialize(self, model, stim_single_cell):
        self.cardiac_model = model
        self.stim_sequence = stim_single_cell

    def collect_kernel_args(self, kernel_args_order):
        kernel_args = []
        for name in kernel_args_order:
            if name in self.cardiac_model.state_vars:
                kernel_args.append(getattr(self.cardiac_model, f"init_{name}"))
                continue
            
            if name in self.cardiac_model.state_pars:
                kernel_args.append(getattr(self.cardiac_model, name))
                continue

            raise ValueError(f"Single-cell kernel argument {name} not found in state variables or parameters.")
        
        return kernel_args
    
    def run(self, history=True):
        self.kernel_generator.history = history
        kernel, kernel_args_order = self.kernel_generator.generate_model_kernel(self.cardiac_model)
        kernel_args = self.collect_kernel_args(kernel_args_order)

        dt = self.stim_sequence.dt
        stim_current = self.stim_sequence.stim_current

        u = self.cardiac_model.init_u
        
        if history:
            self.times = dt * np.arange(len(stim_current))
            self.stim_current = stim_current
            self.u_history = np.zeros(len(stim_current), dtype=np.float32)
            self.u_history[0] = u
            kernel_args = [self.u_history] + kernel_args

        state_vals = kernel(stim_current, dt, u, *kernel_args)

        state_vars = dict(zip(self.cardiac_model.state_vars, state_vals))
        return state_vars