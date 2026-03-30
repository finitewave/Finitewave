import re
import warnings
import textwrap


class IonicKernelGenerator:    
    """
    Base generator for model ionic kernels.

    Attributes
    ----------
    arrays : list
        Names passed as array arguments (e.g., u, v, gating variables, current fields)
    scalars : list
        Names passed as scalar arguments (e.g., parameters)
    observers : list
        List of dicts: {"name": <arg_name>, "expr": <code>}
        where expr is injected at the end of the per-cell loop body.

    Notes
    -----
    - Observers are advanced instrumentation;
    - `expr` must be numba-friendly and race-safe.
    - Better no dynamic append / allocation in parallel kernels.
    """

    def __init__(self):
        self.arrays = []
        self.scalars = []
        self.args_order = [] # does not include rhs, indexes, dt, step and observers
        self.observers = []

        self.names = ["u"]
        self.param_fields = set()

    def _indexing(self, name):
        if name in self.arrays:
            return f"{name}{self._raw_indexing()}"
        return name
        
    def _raw_indexing(self):
        return  ".flat[ind]"

    def generate_observers(self) -> tuple:
        ident = re.compile(r"^[A-Za-z_]\w*$")

        if not self.observers:
            return [], ""

        if not isinstance(self.observers, (list, tuple)):
            raise TypeError("observers must be a list of dicts with keys: 'name', 'expr'.")

        seen = set()
        args = []
        lines = []

        for idx, obs in enumerate(self.observers):
            if not isinstance(obs, dict):
                raise TypeError(f"Observer #{idx} must be a dict.")
            if "name" not in obs or "expr" not in obs:
                raise ValueError(f"Observer #{idx} must have keys 'name' and 'expr'.")

            name = obs["name"]
            expr = obs["expr"]

            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Observer #{idx}: 'name' must be a non-empty string.")
            name = name.strip()

            if not ident.match(name):
                raise ValueError(
                    f"Observer #{idx}: invalid name '{name}'. Must be a valid Python identifier."
                )
            
            if name in set(self.kernel_base_args()):
                raise ValueError(f"Observer name '{name}' collides with kernel arg name.")

            if name in seen:
                raise ValueError(f"Duplicate observer name '{name}'.")
            seen.add(name)

            if not isinstance(expr, str) or not expr.strip():
                raise ValueError(f"Observer '{name}': 'expr' must be a non-empty string.")
            expr = expr.strip()

            if "append(" in expr or ".append(" in expr:
                warnings.warn(
                    f"Observer '{name}': 'append' in expr is unsafe in numba-parallel kernels. "
                    f"Use preallocated arrays like {name}[step, ...] = value.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if "import " in expr:
                warnings.warn(
                    f"Observer '{name}': imports in expr are not allowed/expected.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if name not in expr:
                warnings.warn(
                    f"Observer '{name}': expr does not reference its output buffer '{name}'. "
                    f"Did you forget to write into it?",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if "=" not in expr and not expr.lstrip().startswith("if "):
                warnings.warn(
                    f"Observer '{name}': expr has no '='; it may not store anything.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            args.append(name)
            lines.append(expr)

        return args, "\n        ".join(lines) # 8 spaces for indentation

    def kernel_func_name(self) -> str:
        return "ionic_kernel"

    def kernel_base_args(self) -> list[str]:
        # common arguments: output, indexes, dt, step
        args = ["rhs", "indexes", "dt", "step"]
        args.extend(self.args_order)

        return args

    def generate_loop_header(self) -> str:
        loop_header = """
    for i in prange(len(indexes)):
        idx = indexes[i]
        """
        return textwrap.dedent(loop_header).strip()

    def generate_body(self) -> str:
        """
        Subclasses must override this to generate the per-cell body BEFORE observers.
        Must end with state updates.
        """
        raise NotImplementedError

    def generate_cpu_numba(self) -> str:
        args = ", ".join(self.kernel_base_args())
        loop = self.generate_loop_header()

        # double check required body args
        missing = set(self.args_order)  - set(self.arrays) - set(self.scalars)
        if missing:
            raise ValueError(f"Kernel args missing: {sorted(missing)}")
        body = self.generate_body()
        
        obs_args, obs = self.generate_observers()

        src =f"""
        @njit(parallel=True, fastmath=True)
        def {self.kernel_func_name()}({args + (', ' + ', '.join(obs_args) if obs_args else '')}):
        {loop}
        {body}
        {obs}
        """
        return textwrap.dedent(src).strip()
    

class FooKernelGenerator(IonicKernelGenerator):
    def __init__(self):
        super().__init__()
        self.arrays = ["u", "v"]
        self.scalars = ["a", "b"]
        self.args_order = ["u", "v", "a", "b"]

    def generate_body(self) -> str:
        body = """
            u_new = u.flat[idx] + a * v.flat[idx]
            v.flat[idx] += b * u.flat[idx]
            rhs.flat[idx] = u_new
        """
        return textwrap.dedent(body).strip()


kernel_gen = FooKernelGenerator()
kernel = kernel_gen.generate_cpu_numba()
print(kernel)