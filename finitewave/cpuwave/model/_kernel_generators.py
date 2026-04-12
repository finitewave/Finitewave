import textwrap
import numpy as np

from finitewave.cpuwave.model._kernel_builder import _build_cached
from finitewave.core.model.kernel_generator import KernelGenerator


class IonicKernelGenerator(KernelGenerator):
    def __init__(self, kernel_func_name="ionic_kernel"):
        super().__init__()
        self.kernel_func_name = kernel_func_name
        self.common_args = ["rhs", "u", "indexes", "dt"]
    
    def _update_indexing(self, name, arrays):
        """
        Indexing that is used for updating state variables with new values.
        """
        if name in arrays:
            return f"{name}.flat[idx]"
        return name
        
    def _assign_indexing(self, name, arrays):
        """
        Indexing that is used to assign state variables to temporary (old) value.
        """
        if name in arrays:
            return f"{name}.flat[idx]"
        return name
    
    def _generate_args(self, vars, base="", suffix="", exclude=[]):

        for var in vars:
            if var in exclude:
                continue

            if base == "":
                base += f"{var}{suffix}"
                continue

            base += f", {var}{suffix}"

        return base
    
    def _add_indent(self, code, indent):
        if code.strip() == "":
            return ""
        return textwrap.indent("\n" + textwrap.dedent(code).strip(), " " * indent)
    
    def generate_input_setup(self, arrays, scalars, indent):
        return ""

    def generate_loop(self, indent):
        """
        Returns
        -------
        str
            The header for the loop that iterates over the indexes.
        """
        loop = """\
            for i in range(len(indexes)):
                idx = indexes[i]
        """
        return self._add_indent(loop, indent)
    
    def generate_assignments(self, arrays, indent):
        """
        Returns
        -------
        str
            Code for assigning arrays values to temporary (old) variables.
        """
        asign_vars = "\n".join(f"{var}_old = {self._assign_indexing(var, arrays)}" for var in arrays)
        return self._add_indent(asign_vars, indent)
    
    def generate_update_states(self, state_vars, arrays, indent):
        """
        Returns
        -------
        str
            Code for updating state variables with new values after the step.
        """
        update_vars = [var for var in state_vars if var != "u"]
        update_vars += ["rhs"]

        update_vars = "\n".join(f"{self._update_indexing(var, arrays)} = {var}_new" for var in update_vars)
        return self._add_indent(update_vars, indent)
    
    def generate_output(self, output_args, indent):
        """
        Returns
        -------
        str
            Code for writing the updated state variables back to the output arrays.
        """
        output = "return " + ", ".join(f"{var}" for var in output_args) if output_args else ""
        return self._add_indent(output, indent)
    
    def generate_step_func(self, step_func, arrays, scalars, state_vars, obs_args):

        func_name, body = self.extract_func_body(step_func)

        input_args = "dt, u"
        input_args = self._generate_args(arrays, input_args, exclude=self.common_args)
        input_args = self._generate_args(scalars, input_args, exclude=self.common_args)


        output_args = "rhs, " + ", ".join(f"{var}_new" for var in state_vars if var not in self.common_args)
        output_args += (", " + ", ".join(obs_args) if obs_args else "")

        output_args = "rhs"
        output_args = self._generate_args(state_vars, output_args, suffix="_new", exclude=self.common_args)
        output_args = self._generate_args(obs_args, output_args, exclude=self.common_args)

        body = self._add_indent(body, 16)

        body_func = f"""\
            @njit(fastmath=True)
            def {func_name}({input_args}):
                {body}
                return {output_args}
        """

        body_func = textwrap.dedent(body_func).strip()

        signature_inputs = "dt, u_old"
        signature_inputs = self._generate_args(arrays, signature_inputs, suffix="_old", exclude=self.common_args)
        signature_inputs = self._generate_args(scalars, signature_inputs, exclude=self.common_args)
    
        signature_returns = "rhs_new"
        signature_returns = self._generate_args(state_vars, signature_returns, suffix="_new", exclude=self.common_args)
        signature_returns = self._generate_args(obs_args, signature_returns, exclude=self.common_args)

        func_signature = f"{signature_returns} = {func_name}({signature_inputs})"
        return func_name, func_signature, body_func
    
    def generate_observers(self, observers, state_vars, indent):
        if len(observers) == 0:
            return "", set(), set()   

        names = set()
        expr_lines = []
        expr_args = []
        kernel_args = []

        for obs in observers:
            
            name, expr, e_args, k_args = obs.generate(state_vars)
            
            if name in set(kernel_args):
                raise ValueError(f"Observer name '{name}' collides with kernel arg name.")

            if name in names:
                raise ValueError(f"Duplicate observer name '{name}'.")

            names.add(name)
            expr_lines.append(expr)
            expr_args.extend(e_args)
            kernel_args.extend(k_args)

        expr_lines = self._add_indent("\n".join(expr_lines), indent)

        return expr_lines, set(expr_args), set(kernel_args)

    def generate_body(self, step_func, arrays, scalars, state_vars, observers=[], output_args=[]):
        """Generate numba CPU kernel."""
        input_setup = self.generate_input_setup(arrays, scalars, indent=16)
        loop = self.generate_loop(indent=16)
        asign_vars = self.generate_assignments(arrays, indent=20)
        update_states = self.generate_update_states(state_vars, arrays, indent=20)
        obs, obs_args, obs_kernel_args = self.generate_observers(observers, state_vars, indent=20)
        output = self.generate_output(output_args, indent=16)

        kernel_args = list(arrays) + list(scalars) + list(obs_kernel_args)
        kernel_args = list(set(kernel_args) - set(self.common_args))
        kernel_args_str = ", ".join(self.common_args + kernel_args)

        step_res = self.generate_step_func(step_func, arrays, scalars, state_vars, obs_args)
        step_func_name, step_func_signature, step_func_body = step_res

        kernel_func =f"""\
            @njit(parallel=True, fastmath=True)
            def {self.kernel_func_name}({kernel_args_str}):
                {input_setup}
                {loop}
                    {asign_vars}\n
                    {step_func_signature}
                    {update_states}
                    {obs}
                {output}
            """
        kernel_func = textwrap.dedent(kernel_func).strip()

        # print("\nGenerated kernel function:")
        # print(kernel_func)
        return step_func_name, step_func_body, kernel_func, kernel_args
    
    def generate(self, step_func, arrays, scalars, state_vars, model_func, observers=[], output_args=[]):
        
        res = self.generate_body(step_func, arrays, scalars, state_vars, observers, output_args)
        step_func_name, step_func_body, kernel_func, kernel_args = res

        sorted_model_func = tuple(sorted(model_func.items(), key=lambda kv: kv[0]))

        step_func = _build_cached(
            step_func_name,
            step_func_body,
            sorted_model_func,
        )
        
        sorted_model_func += ((step_func_name, step_func),)

        kernel_func = _build_cached(
            self.kernel_func_name,
            kernel_func,
            sorted_model_func,
        )

        return kernel_func, list(kernel_args)

    def generate_model_kernel(self, model, step_func, model_func, observers=[], output_args=[]):
        arrays, scalars = self._collect_arrays(model)
        arrays = ["rhs"] + arrays
        state_vars = model.state_vars

        return self.generate(step_func, arrays, scalars, state_vars, model_func, observers, output_args)

    def _collect_arrays(self, model):
        arrays = list(model.state_vars)
        scalars = []

        for param in model.state_pars:
            if np.isscalar(getattr(model, f"{param}")):
                scalars.append(param)
            else:
                arrays.append(param)

        return arrays, scalars


class PrepacingKernelGenerator(IonicKernelGenerator):
    def __init__(self, kernel_func_name="prepacing_kernel"):
        super().__init__(kernel_func_name)
        self.common_args = ["u_pacing", "stim_values", "dt", "u"]

    def _assign_indexing(self, name, arrays):
        if name in arrays:
            return f"{name}.flat[idx-1]"
        return name
        
    def _update_indexing(self, name, arrays):
        if name in arrays:
            return f"{name}.flat[idx]"
        return name
    
    def generate_loop(self, indent) -> str:
        """
        Returns
        -------
        str
            The header for the loop that iterates over the indexes.
        """
        loop = """\
            u_pacing[0] = u
            for idx in range(1, len(stim_values)):
                u_old = u + dt * stim_values.flat[idx-1]
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
        update_vars += "\nu = u_old + dt * rhs_new"
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
    
    def generate_model_kernel(self, model, step_func, model_func):
        arrays = []
        scalars = list(model.state_vars) + list(model.state_pars)
        state_vars = model.state_vars
        output_args = state_vars
        return self.generate(step_func, arrays, scalars, state_vars, model_func,
                             output_args=output_args)


class StateCollectingKernelGenerator(IonicKernelGenerator):
    def __init__(self, kernel_func_name="phase_states_kernel"):
        super().__init__(kernel_func_name)
        self.common_args = ["dt", "t_max", "step", "u"]
    
    def _assign_indexing(self, name, arrays):
        return name
        
    def _update_indexing(self, name, arrays):
        if name in arrays:
            return f"{name}_states.flat[idx_step]"
        return name
    
    def generate_input_setup(self, arrays, scalars, indent):
        assign_init_vals = "\n".join(f"{var}_states = {var} * np.ones(int(t_max / (dt * step)), dtype=np.float64)" for var in arrays)
        return self._add_indent(assign_init_vals, indent)

    def generate_loop(self, indent) -> str:
        """
        Returns
        -------
        str
            The header for the loop that iterates over the indexes.
        """
        
        loop = f"""\
            n_iters = int(t_max / dt)
            for idx in range(1, n_iters):
        """
        return self._add_indent(loop, indent)

    def generate_update_states(self, state_vars, arrays, indent) -> str:
        """
        Returns
        -------
        str
            Code for updating state variables with new values after the step.
        """
        assign_vars = "\n".join(f"{var} = {var}_new" for var in state_vars if var != "u")
        assign_vars = self._add_indent(assign_vars, 12)
        update_states = "\n".join(f"{self._update_indexing(var, arrays)} = {var}" for var in arrays)
        update_states = self._add_indent(update_states, 12)

        update_vars = f"""\
            {assign_vars}
            u = u_old + dt * rhs_new

            idx_step = idx % step
            {update_states}
        """
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

        output += ", ".join(f"{var}_states" for var in output_args)
        return self._add_indent(output, indent)

    def generate_model_kernel(self, model, step_func, model_func):
        arrays = list(model.state_vars)
        scalars = list(model.state_pars)
        state_vars = model.state_vars
        output_args = state_vars
        return self.generate(step_func, arrays, scalars, state_vars, model_func,
                             output_args=output_args)


# def ionic_step(dt, u, v, a, k, eps, mu1, mu2):
#     dv = (- (eps + (mu1 * v) / (mu2 + u)) * (v + k * u * (u - a - 1.)))
#     rhs = -k * u * (u - a) * (u - 1) - u * v
#     v_new = v + dt * dv
#     return rhs, v_new

# state_vars = ["u", "v", "w"]
# arrays = []
# parameters = ["a", "k", "eps", "mu1", "mu2"]
# scalars = parameters + ["v", "w", "u"]
# output_args = arrays
# # obs_args = ["dv"]

# # obs = Observer(
# #     target="dv_obs",
# #     expr="if idx == 100: dv_obs[ind_obs] = dv",
# #     expr_args=["dv"],
# #     kernel_args=["ind_obs"],
# # )

# prepacing_generator = PrepacingKernelGenerator()
# res_prepacing = prepacing_generator.generate(ionic_step, arrays, scalars, state_vars, {},
#                                              output_args=state_vars)

# print("Prepacing kernel:")
# print(res_prepacing)

# state_vars = ["u", "v", "w"]
# arrays = ["u", "v", "w"]
# parameters = ["a", "k", "eps", "mu1", "mu2"]
# scalars = parameters
# output_args = arrays

# state_generator = PhaseStatesKernelGenerator()
# res = state_generator.generate(ionic_step, arrays, scalars, state_vars, {},
#                                output_args=output_args)
# print("Phase states kernel:")
# print(res)
