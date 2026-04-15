import numpy as np
import textwrap
from finitewave.core.model.kernel_generator import KernelGenerator
from finitewave.mlxwave.model.kernel._func_builder import wrap_mlx_func, build_func


class IonicMlxGenerator(KernelGenerator):
    """
    Class for generating numba CPU kernel function to update state variables
    and compute the ionic current (rhs) of the cardiac model.

    The kernel function is generated based on a user-defined step function that
    computes the new state variables and rhs based on the current state and parameters.
    The generator handles the creation of the loop over the spatial indexes,
    the assignment of old values, the update of state variables,
    and the integration of observer computations.

    Attributes
    ----------
    kernel_func_name : str
        The name to use for the generated kernel function.
    common_args : list of str
        List of argument names that are common to all models and 
        should be included in the kernel function signature.
        
    """
    def __init__(self, kernel_func_name="ionic_kernel"):
        """
        Parameters
        ----------
        kernel_func_name : str, optional
            The name to use for the generated kernel function (default: "ionic_kernel").
        """
        super().__init__()
        self.kernel_func_name = kernel_func_name
        self.common_args = ["dt", "u"]
    
    def _generate_args(self, vars, base="", suffix="", exclude=[]):
        """
        Helper function to generate argument strings for the step function
        and kernel function.
        
        Parameters
        ----------
        vars : list of str
            List of variable names to include in the argument string.
        base : str, optional
            Initial string to which the variables will be appended.
        suffix : str, optional
            Suffix to append to each variable name (e.g., "_new" or "_old").
        exclude : list of str, optional
            List of variable names to exclude from the argument string.
        
        Returns
        -------
        str
            A string of the form "base, var1{suffix}, var2{suffix}, ..."
            for all vars not in exclude.
        """

        for var in vars:
            if var in exclude:
                continue

            if base == "":
                base += f"{var}{suffix}"
                continue

            base += f", {var}{suffix}"

        return base
    
    def _add_indent(self, code, indent):
        """
        Helper function to add indentation to a block of code.

        Parameters
        ----------
        code : str
            The code block to indent.
        indent : int
            The number of spaces to use for indentation.

        Returns
        -------
        str
            The indented code block.
        """
        if code.strip() == "":
            return ""
        return textwrap.indent("\n" + textwrap.dedent(code).strip(), " " * indent)
    
    def generate_input_setup(self, arrays, scalars, indent):
        """
        Generate code for setting up input variables before the loop."""
        return ""
    
    def generate_output(self, output_args, indent):
        """
        Generate return statement for the kernel function based on the specified output arguments.

        Parameters
        ----------
        output_args : list of str
            List of argument names to include in the output.
        indent : int
            The number of spaces to use for indentation.

        Returns
        -------
        str
            Code for return statement of the kernel function.
        """
        output = "return " + ", ".join(f"{var}" for var in output_args) if output_args else ""
        return self._add_indent(output, indent)
    
    def generate_step_func(self, step_func, arrays, scalars, state_vars, obs_args):
        """
        Generate the step function that computes the new state variables and rhs based on the current state and parameters.
        
        Parameters
        ----------
        step_func : function
            The user-defined function that computes the new state variables and rhs.
        arrays : list of str
            List of array variable names used in the step function.
        scalars : list of str
            List of scalar variable names used in the step function.
        state_vars : list of str
            List of state variable names.
        obs_args : list of str
            List of observer extra arguments which should be returned from
            the step function.
            
        Returns
        -------
        func_name : str
            The name of the generated step function.
        func_signature : str
            The signature of the generated step function.
        func_body : str
            The body of the generated step function.
        """

        func_name, body = self.extract_func_body(step_func)

        input_args = "dt"
        input_args = self._generate_args(arrays, input_args, exclude=["rhs"])
        input_args = self._generate_args(scalars, input_args)
                                                                                                                                                    
        output_args = "rhs"
        output_args = self._generate_args(state_vars, output_args, suffix="_new", exclude=["u"])
        output_args = self._generate_args(obs_args, output_args, exclude=state_vars)

        body = self._add_indent(body, 16)

        func_body = f"""\
            def {func_name}({input_args}):
                {body}
                return {output_args}
        """

        func_body = textwrap.dedent(func_body).strip()

        print("\nGenerated step function:")
        print(func_body)

        signature_inputs = input_args
    
        signature_returns = "rhs"
        signature_returns = self._generate_args(state_vars, signature_returns, suffix="", exclude=["u"])
        signature_returns = self._generate_args(obs_args, signature_returns)

        func_signature = f"{signature_returns} = {func_name}({signature_inputs})"
        return func_name, func_signature, func_body
    
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
        obs, obs_args, obs_kernel_args = self.generate_observers(observers, state_vars, indent=20)
        output = self.generate_output(output_args, indent=16)

        kernel_args = list(arrays) + list(scalars) + list(obs_kernel_args)
        kernel_args = [var for var in kernel_args if var not in self.common_args]
        kernel_args_str = ", ".join(self.common_args + kernel_args)

        step_res = self.generate_step_func(step_func, arrays, scalars, state_vars, obs_args)
        step_func_name, step_func_signature, step_func_body = step_res

        kernel_func =f"""\
            def {self.kernel_func_name}({kernel_args_str}):
                {input_setup}
                {step_func_signature}
                {obs}
                {output}
            """
        kernel_func = textwrap.dedent(kernel_func).strip()

        print("\nGenerated kernel function:")
        print(kernel_func)
        return step_func_name, step_func_body, kernel_func, kernel_args
    
    def generate_kernel(self, ops, arrays, scalars, state_vars, observers=[], output_args=[]):
        """
        Generate the kernel function by combining the step function, model function, and observers.
        
        Parameters
        ----------
        step_func : function
            The user-defined function that computes the new state variables and rhs.
        model_func : dict
            A dictionary of functions that are called within the step function.
        arrays : list of str
            List of array variable names used in the step function.
        scalars : list of str
            List of scalar variable names used in the step function.
        state_vars : list of str
            List of state variable names.
        observers : list of Observer, optional
            List of Observer instances that define additional computations to be performed during the kernel execution.
        output_args : list of str, optional
            List of argument names to include in the output of the kernel function.
            
        Returns
        -------
        kernel_func : function
            The jitted kernel function that can be used for simulation.
        kernel_args : list of str
            List of argument names that the kernel function expects.
        """
        step_func = ops.ionic_step
        res = self.generate_body(step_func, arrays, scalars, state_vars, observers, output_args)
        step_func_name, step_func_body, kernel_func, kernel_args = res
        
        model_func, glb_funcs = wrap_mlx_func(ops)
        
        # sort and make it hashable to ensure consistent ordering for caching
        glb_funcs = tuple(sorted(glb_funcs.items()))
        model_func = tuple(sorted(model_func.items()))

        # print("\nGenerated step function:")
        # print(step_func_body)

        step_func = build_func(
            step_func_name,
            step_func_body,
            glb_funcs,
            model_func
        )
        
        model_func += ((step_func_name, step_func),)

        kernel_func = build_func(
            self.kernel_func_name,
            kernel_func,
            glb_funcs,
            model_func
        )

        return kernel_func, list(kernel_args)

    def generate_model_kernel(self, model):
        """Generate the ionic kernel function for a given cardiac model instance.
        
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
        observers = model.observers
        arrays, scalars = self._collect_model_arrays(model)
        # arrays = ["rhs"] + arrays
        state_vars = model.state_vars
        output_args = ["rhs"] + [var for var in state_vars if var != "u"]

        kernel_func, kernel_args = self.generate_kernel(model.ops,
                                                        arrays, scalars,
                                                        state_vars, observers,
                                                        output_args)
        return kernel_func, kernel_args

    def _collect_model_arrays(self, model):
        """Collect array and scalar variable names from the model.
        
        Parameters
        ----------
        model : CardiacModel
            The cardiac model instance from which to collect variable names.
            
        Returns
        -------
        arrays : list of str
            List of variable names that correspond to arrays (state variables).
        scalars : list of str
            List of variable names that correspond to scalars (parameters).
        """
        arrays = list(model.state_vars)
        scalars = []

        for param in model.state_pars:
            if np.isscalar(getattr(model, f"{param}")):
                scalars.append(param)
            else:
                arrays.append(param)

        return arrays, scalars



# from finitewave.mlxwave.model.kernel._load_ops import load_ops


# ops = load_ops("aliev_panfilov")

# model_func = {}
# arrays = ops.get_variables().keys()
# scalars = ops.get_parameters().keys()
# state_vars = ops.get_variables().keys()
# output_args = ["rhs"] + [var for var in state_vars if var != "u"]


# generator = IonicMlxGenerator()
# # generator.generate_body(ionic_step, arrays, scalars, state_vars, output_args=output_args)
# kernel_func, kernel_args = generator.generate_kernel(ops, arrays, scalars, state_vars, output_args=output_args)


# import mlx.core as mx

# u = mx.zeros(10, dtype=mx.float32)
# v = mx.zeros(10, dtype=mx.float32)

# a = 0.1
# k = 8.0
# eps = 0.01
# mu1 = 0.2
# mu2 = 0.3
# dt = 0.01

# # rhs, v = kernel_func(dt, u, v, a, k, eps, mu1, mu2)

# def calc_rhs(u, v, a, k):
#     return -k * u * (u - a) * (u - 1) - u * v

# print(ops.calc_rhs(u, v, a, k))

# rhs = calc_rhs(u, v, a, k)