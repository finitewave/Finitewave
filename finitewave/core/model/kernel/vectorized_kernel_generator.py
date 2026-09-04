import re
import warnings
import textwrap
import ast
import inspect


class VectorizedKernelGenerator:    
    """
    Base generator for model ionic kernels. 

    Attributes
    ----------
    kernel_func_name : str
        Name of the generated kernel function.
    common_args : list
        Names of common arguments passed to all kernels (e.g., rhs, indexes, dt, step)
    arrays : list
        Names passed as array arguments (e.g., u, gating variables, current fields)
    scalars : list
        Names passed as scalar arguments (e.g., parameters)
    model_args : list
        Names of all model-specific arguments (`arrays` + `scalars`): `step` function signature.
    state_vars : list
        Names of state variables that are updated in the step function (e.g., u, v).
    output_args : list
        Names of arguments that should be returned.
    observers : list
        List of dicts: {"name": <arg_name>, "expr": <code>}
        where expr is injected at the end of the per-cell loop body.
    body : str
        The body of the kernel function, executed for each cell/index.

    Notes
    -----
    - Observers are advanced instrumentation;
    - `expr` must be numba-friendly and race-safe.
    - Better no dynamic append / allocation in parallel kernels.
    """

    def __init__(self):
        self.kernel_func_name = ""
        self.common_args = []
        self.arrays = []
        self.scalars = []
        self.model_args = []
        # self.state_vars = []
        # self.observers = []
        # self.output_args = []
        self.body = ""

    def generate_body(self) -> str:
        """
        The body of the kernel function, executed for each cell/index.
        """
        raise NotImplementedError
    
    def generate(self) -> str:
        """
        Generates the complete kernel function source code as a string.
        """
        raise NotImplementedError
    
    def check_args(self, model_args, arrays, scalars):
        """
        Validates that all required args are included in the arrays and scalars.
        """
        missing = set(model_args) - set(arrays) - set(scalars)

        if missing:
            raise ValueError(f"Kernel args missing: {sorted(missing)}")
        
    def extract_func_body(self, func) -> str:
        """
        Extracts the body of a function as a string, removing the docstring,
        function definition line, and return statements.

        Parameters
        ----------
        func : function
            The function whose body is to be extracted.

        Returns
        -------
        str
            The source code of the function body.
        """
        func_name = func.__name__
        src = inspect.getsource(func)
        src = textwrap.dedent(src)
        tree = ast.parse(src)

        func_node = tree.body[0]

        new_body = []
        return_count = 0
        for node in func_node.body:
            # Remove docstring
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            
            # Remove return statements and count them to ensure there's only one (if any)
            if isinstance(node, ast.Return):
                return_count += 1
                if return_count > 1:
                    raise ValueError("Multiple return statements are not supported in kernel functions.")
                continue

            new_body.append(node)

        module = ast.Module(body=new_body, type_ignores=[])
        func_body = ast.unparse(module)
        func_body = textwrap.dedent(func_body)
        return func_name, func_body

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