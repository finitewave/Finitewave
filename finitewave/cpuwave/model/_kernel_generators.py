import textwrap
import re
import warnings
import numpy as np

from finitewave.core.model.kernel_generator import KernelGenerator


class StepKernelGenerator(KernelGenerator):
    def __init__(self):
        super().__init__()
        self.kernel_func_name = "ionic_kernel"
        self.common_args = ["rhs", "indexes", "dt"]
    
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
    
    def generate_loop(self) -> str:
        """
        Returns
        -------
        str
            The header for the loop that iterates over the indexes.
        """
        loop = """\
            for i in prange(len(indexes)):
                idx = indexes[i]
        """
        return textwrap.dedent(loop).strip()
    
    def generate_assignments(self, arrays) -> str:
        """
        Returns
        -------
        str
            Code for assigning arrays values to temporary (old) variables.
        """
        asign_vars = "\n".join(f"{var}_old = {self._assign_indexing(var, arrays)}" for var in arrays)
        return textwrap.dedent(asign_vars).strip()
    0
    def generate_update_states(self, state_vars, arrays) -> str:
        """
        Returns
        -------
        str
            Code for updating state variables with new values after the step.
        """
        update_vars = "\n".join(f"{self._update_indexing(var, arrays)} = {var}_new" for var in state_vars)
        return textwrap.dedent(update_vars).strip()
    
    def generate_output(self, output_args) -> str:
        """
        Returns
        -------
        str
            Code for writing the updated state variables back to the output arrays.
        """
        output = "return " + ", ".join(f"{var}" for var in output_args) if output_args else ""
        return output
    
    def generate_step(self, body, arrays, scalars, obs_args, state_vars) -> str:
        input_args = ", ".join(f"{var}_old" for var in arrays)
        input_args += (", " + ", ".join(f"{var}" for var in scalars))
        output_args = "rhs_new, " + ", ".join(f"{var}_new" for var in state_vars)
        output_args += (", " + ", ".join(obs_args) if obs_args else "")

        body = textwrap.indent("\n" + textwrap.dedent(body).strip(), " " * (12 + 4))

        body_func = f"""\
            @njit(fastmath=True)
            def step({input_args}):
                {body}
                return {output_args}
        """

        func_signature = f"{output_args} = step({input_args})"
        return func_signature, textwrap.dedent(body_func).strip()
    
    def generate_body(self, step_func):
        body = self.extract_func_body(step_func)
        return body
    
    def generate_observers(self, observers, kernel_args):
        if len(observers) == 0:
            return [], ""
        

        seen = set()
        args = []
        lines = []

        for obs in observers:
            
            name, expr = obs.generate()
            
            if name in set(kernel_args):
                raise ValueError(f"Observer name '{name}' collides with kernel arg name.")

            if name in seen:
                raise ValueError(f"Duplicate observer name '{name}'.")

            seen.add(name)

            args.append(name)
            lines.append(expr)

        return args, "\n".join(lines)

    def generate_kernel(self, step_func, arrays, scalars, state_vars, observers=[], output_args=[]) -> str:
        """Generate numba CPU kernel."""

        model_args = arrays + scalars

        loop = self.generate_loop()
        asign_vars = self.generate_assignments(arrays)
        update_states = self.generate_update_states(state_vars, arrays)
        obs_args, obs = self.generate_observers(observers)
        body = self.generate_body(step_func)
        output = self.generate_output(output_args)
        
        # add empty line to ignore indentation in src
        # remove original indentation and add new one for the whole block
        loop = textwrap.indent("\n" + textwrap.dedent(loop).strip(), " " * (12 + 4))
        asign_vars = textwrap.indent("\n" + textwrap.dedent(asign_vars).strip(), " " * (12 + 8))
        body = textwrap.indent("\n" + textwrap.dedent(body).strip(), " " * (12 + 8))
        update_states = textwrap.indent("\n" + textwrap.dedent(update_states).strip(), " " * (12 + 8))
        obs = textwrap.indent("\n" + textwrap.dedent(obs).strip(), " " * (12 + 8)) if obs else ""
        output = textwrap.indent("\n" + textwrap.dedent(output).strip(), " " * (12 + 4)) if output else ""

        args = ", ".join(model_args)
        args += (", " + ", ".join(obs_args) if obs_args else "")

        
        body_func_name, body_func = self.generate_step(body, arrays, scalars, obs_args, state_vars)

        src =f"""\
            @njit(parallel=True, fastmath=True)
            def {self.kernel_func_name}({args + (', ' + ', '.join(obs_args) if obs_args else '')}):
                {loop}
                    {asign_vars}
                    {body_func_name}
                    {obs}
                    {update_states}
                {output}
            """
        return textwrap.dedent(src).strip()
    

class Observer:
    """
    Class representing an observer that can be added to the kernel.
    
    Attributes
    ----------
    target : str
        The name of the output variable that the observer writes to.
    expr : str
        The expression that computes the observer's value.
    expr_args : list of str, optional
        The names of the variables used in the observer's expression.
    kernel_args : list of str, optional
        The names of the kernel arguments that the observer depends on.
    """
    def __init__(self, target, expr, expr_args, kernel_args=None):
        """
        Parameters
        ----------
        target : str
            The name of the output variable that the observer writes to.
        expr : str
            The expression that computes the observer's value.
        expr_args : list of str, optional
            The names of the variables used in the observer's expression.
        kernel_args : list of str, optional
            The names of the kernel arguments that the observer depends on.
            If None, it is assumed the observer depends on existing kernel args.
        """
        self.target = target
        self.expr = expr
        self.expr_args = expr_args
        self.kernel_args = kernel_args

    def generate(self) -> tuple[str, str, list, list]:
        """
        Validates and generates code for observers.
        """
        ident = re.compile(r"^[A-Za-z_]\w*$")

        if self.target is None or self.expr is None:
            raise ValueError("Observer must have 'target' and 'expr' defined.")

        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError(f"Observer #{self.target}: 'target' must be a non-empty string.")

        target = self.target.strip()

        if not ident.match(target):
            raise ValueError(
                f"Observer #{target}: invalid name '{target}'. Must be a valid Python identifier."
            )
        
        if not isinstance(self.expr, str) or not self.expr.strip():
            raise ValueError(f"Observer '{target}': 'expr' must be a non-empty string.")

        expr = self.expr.strip()

        if "append(" in expr or ".append(" in expr:
            warnings.warn(
                f"Observer '{target}': 'append' in expr is unsafe in numba-parallel kernels. "
                f"Use preallocated arrays like {target}[step, ...] = value.",
                RuntimeWarning,
                stacklevel=2,
            )
        if "import " in expr:
            warnings.warn(
                f"Observer '{target}': imports in expr are not allowed/expected.",
                RuntimeWarning,
                stacklevel=2,
            )
        if target not in expr:
            warnings.warn(
                f"Observer '{target}': expr does not reference its output buffer '{target}'. "
                f"Did you forget to write into it?",
                RuntimeWarning,
                stacklevel=2,
            )
        if "=" not in expr and not expr.lstrip().startswith("if "):
            warnings.warn(
                f"Observer '{target}': expr has no '='; it may not store anything.",
                RuntimeWarning,
                stacklevel=2,
            )

        return target, expr, self.expr_args, self.kernel_args
    

class SingleCellKernelGenerator(StepKernelGenerator):
    def __init__(self):
        super().__init__()
        self.kernel_func_name = "single_cell_kernel"
        self.arrays = ["u", "stim_values"]
        self.common_args = ["u", "stim_values", "dt"]
        self.output_args = ["u"]
        self.state_vars = ["u", "v"]

    def _assign_indexing(self, name):
        """Indexing that is used to assign state variables to temporary (old) value.

        Parameters
        ----------
        name : str
            Name of the variable to index.
        """
        if name in self.arrays:
            return f"{name}.flat[idx-1]"
        return name
        
    def _update_indexing(self, name):
        """Indexing that is used for updating state variables with new values.

        Parameters
        ----------
        name : str
            Name of the variable to index.
        """
        if name in self.arrays:
            return f"{name}.flat[idx]"
        return name
    
    def generate_loop(self) -> str:
        """
        Returns
        -------
        str
            The header for the loop that iterates over the indexes.
        """
        loop = """\
            for idx in prange(1, len(stim_values)):
                u.flat[idx-1] = u.flat[idx-1] + dt * stim_values.flat[idx-1]
        """
        return textwrap.dedent(loop).strip()
    
    def generate_body(self, update_states, observers):
        body = super().generate_body()
        body += "\n" + "u_new = u_old + dt * rhs_new"
        print("Generated body:\n", body)
        return body


def step(dt, u, v, a, k, eps, mu1, mu2):
    dv = (- (eps + (mu1 * v) / (mu2 + u)) * (v + k * u * (u - a - 1.)))
    rhs_new = -k * u * (u - a) * (u - 1) - u * v
    v_new = v + dt * dv
    return rhs_new, v_new

observers = ""
state_vars = ["u", "v"]
arrays = state_vars + ["a"]
parameters = ["a", "k", "eps", "mu1", "mu2"]
obs_args = ["a"]

step_generator = StepKernelGenerator()
print(step_generator.generate_kernel(step, arrays, parameters, state_vars, observers, obs_args))
