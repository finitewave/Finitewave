import textwrap
from attrs import inspect
import numpy as np

from .spec import KernelSpec


class ModelAnalyzer:
    """Collect model metadata without selecting an execution backend."""

    def __init__(self):
        pass

    def analyze(self, model):
        step_function = model.ops.ionic_step

        step_name = step_function.__name__

        step_source = textwrap.dedent(
            inspect.getsource(step_function)
        ).strip()
        
        step_args = tuple(
            inspect.signature(step_function).parameters
        )

        arrays = list(model.state_vars)
        scalars = []

        for name in model.state_pars:
            value = getattr(model, name)
            if np.isscalar(value):
                scalars.append(name)
            else:
                arrays.append(name)

        return KernelSpec(
            name="ionic_kernel",
            ops=model.ops,
            step_function=model.ops.ionic_step,
            step_name=step_name,
            step_source=step_source,
            step_args=step_args,
            common_args=tuple(model.common_args),
            arrays=tuple(arrays),
            scalars=tuple(scalars),
            state_vars=tuple(model.state_vars),
            observers=tuple(model.observers),
            output_args=(
                "rhs",
                *(name for name in model.state_vars if name != "u"),
            ),
        )
