import textwrap

from .artifact import KernelArtifact
from .scheme import KernelScheme, remove_decorators


class VectorizedKernelScheme(KernelScheme):
    """Generate a kernel that does not use explicit loops, but vectorized array operations instead."""

    def generate(self, spec):
        kernel_args = self._collect_kernel_arguments(spec)
        kernel_signature = ", ".join(kernel_args)

        step_args = [ # TODO: _collect_step_arguments(spec) should be flixible or individual. 
            name for name in spec.step_args 
        ]
        step_signature = ", ".join(step_args)

        source = f"""
        def {spec.name}({kernel_signature}):
            return {spec.step_name}(
                {step_signature}
            )
        """

        source = textwrap.dedent(source).strip()

        args_names = tuple(
            name
            for name in kernel_args
            if name not in spec.common_args
        )

        return KernelArtifact(
            spec=spec,
            source=source,
            argument_names=args_names,
            step_name=spec.step_name,
            step_source=spec.step_source,
        )