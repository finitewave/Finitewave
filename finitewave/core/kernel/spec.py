class KernelSpec:
    """Backend-independent kernel specs."""

    def __init__(self, name, ops, step_function, step_name, step_source, step_args, 
                 common_args, arrays, scalars, state_vars, observers, output_args):
        self.name = name
        self.ops = ops

        self.step_function = step_function
        self.step_name = step_name
        self.step_source = step_source
        self.step_args = step_args
        self.common_args = common_args
        self.arrays = arrays
        self.scalars = scalars
        self.state_vars = state_vars
        self.observers = observers
        self.output_args = output_args

    @property
    def observer_step_outputs(self):
        args = []

        for observer in self.observers:
            for name in observer.step_outputs:
                if name not in args:
                    args.append(name)

        return tuple(args)

    @property
    def observer_kernel_args(self):
        args = []

        for observer in self.observers:
            for name in observer.kernel_args:
                if name not in args:
                    args.append(name)

        return tuple(args)
