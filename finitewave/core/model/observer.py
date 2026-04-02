import re
import warnings


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
    def __init__(self, target=None, expr=None, expr_args=None, kernel_args=None):
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

    def generate(self, state_vars):
        """
        Validates and generates code for observers.
        """
        target = self.check_target(self.target)
        expr = self.check_expr(self.expr, target)
        extra_args = self.check_args(self.expr_args, state_vars, target)
        
        kernel_args = self.kernel_args if self.kernel_args is not None else []
        kernel_args += [target]
        return target, expr, extra_args, set(kernel_args)

    def check_expr(self, expr, target):
        if expr is None:
            raise ValueError("Observer must have 'expr' defined.")

        if not isinstance(expr, str) or not expr.strip():
            raise ValueError(f"Observer '{target}': 'expr' must be a non-empty string.")

        expr = expr.strip()

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

        return expr
    
    def check_target(self, target):
        ident = re.compile(r"^[A-Za-z_]\w*$")

        if target is None:
            raise ValueError("Observer must have 'target' defined.")

        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"Observer #{target}: 'target' must be a non-empty string.")

        target = target.strip()

        if not ident.match(target):
            raise ValueError(
                f"Observer #{target}: invalid name '{target}'. Must be a valid Python identifier."
            )
        return target

    def check_args(self, expr_args, state_vars, target):
        if expr_args is None:
            return []
        extra_args = set(expr_args) - set(state_vars)
        if len(extra_args) == 0:
            raise ValueError(f"Observer {target}: All expr_args are state variables. Prefer to use VariableTracker instead.")
        return extra_args
