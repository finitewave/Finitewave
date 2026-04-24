from .ionic_numba_kernel import IonicNumbaKernel


class SingleCellNumbaKernel(IonicNumbaKernel):
    def __init__(self, kernel_func_name="prepacing_kernel"):
        super().__init__(kernel_func_name)
        self.common_args = ["stim_values", "dt", "u"]

    @property
    def history(self):
        return self._history
    
    @history.setter
    def history(self, value):
        self._history = value

        if self._history and "u_pacing" not in self.common_args:
            self.common_args += ["u_pacing"]

        elif not self._history and "u_pacing" in self.common_args:
            self.common_args.remove("u_pacing")
   
    def generate_loop(self, indent) -> str:
        """
        Returns
        -------
        str
            The header for the loop that iterates over the indexes.
        """
        loop = """\
            for idx in range(1, len(stim_values)):
                u = u + dt * stim_values.flat[idx-1]
        """
        return self._add_indent(loop, indent)

    def generate_update_states(self, state_vars, arrays, indent) -> str:
        """
        Returns
        -------
        str
            Code for updating state variables with new values after the step.
        """
        update_vars = "\n".join(f"{self._update_indexing(var, arrays)} = {var}_new" for var in state_vars if var != "u")
        update_vars += "\nu = u + dt * rhs_new"
        if self.history:
            update_vars += "\nu_pacing.flat[idx] = u"
        return self._add_indent(update_vars, indent)

    def generate_output(self, output_args, indent) -> str:
        """
        Returns
        -------
        str
            Code for writing the updated state variables back to the output arrays.
        """
        output = "return "
        if len(output_args) == 0:
            return output

        output += ", ".join(output_args)
        return self._add_indent(output, indent)
    
    def generate_model_kernel(self, model):
        """Generate the prepacing kernel for a given cardiac model instance.

        Parameters
        ----------
        model : CardiacModel
            An instance of a cardiac model containing state variables,
            parameters, and step function.

        Returns
        -------
        kernel_func : function
            The generated kernel function that can be used for simulation.
        kernel_args : list of str
            List of argument names that the kernel function expects.
        """
        arrays = []
        scalars = list(model.state_vars) + list(model.state_pars)
        state_vars = model.state_vars
        output_args = state_vars
        ops = model.ops

        return self.generate_kernel(ops, arrays, scalars, state_vars,
                                    output_args=output_args)