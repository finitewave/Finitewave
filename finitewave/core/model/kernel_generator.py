import re
import warnings
import textwrap
import ast
import inspect


class KernelGenerator:    
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
    
    def generate_loop(self) -> str:
        """
        The header for the loop that iterates over the indexes.
        """
        raise NotImplementedError

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
    
    def generate_observers(self) -> tuple:
        """
        Validates and generates code for observers.
        """
        raise NotImplementedError
