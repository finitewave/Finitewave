from dataclasses import dataclass
from typing import Callable

from .spec import KernelSpec


@dataclass
class KernelArtifact:
    """Generated forms of a kernel from source through backend compilation."""

    def __init__(self, spec, source, argument_names, step_name, step_source):
        self.spec = spec
        self.source = source
        self.argument_names = argument_names

        self.step_name = step_name
        self.step_source = step_source

        self.python_step = None
        self.python_function = None
        self.compiled_step = None
        self.compiled_function = None

    @property
    def function(self):
        if self.compiled_function is not None:
            return self.compiled_function
        if self.python_function is not None:
            return self.python_function
        raise RuntimeError("Kernel has not been built.")

    @property
    def is_built(self):
        return self.python_function is not None

    @property
    def is_compiled(self):
        return self.compiled_function is not None

    def replace_source(self, source):
        """Replace generated source and invalidate its executable forms."""
        self.source = source
        self.python_function = None
        self.compiled_function = None
