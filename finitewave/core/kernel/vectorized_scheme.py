import textwrap

from .artifact import KernelArtifact
from .scheme import KernelScheme


class VectorizedKernelScheme(KernelScheme):
    """Generate a kernel that does not use explicit loops, but vectorized array operations instead."""

    def generate(self, spec):
        kernel_args = self._collect_kernel_arguments(spec)
        kernel_signature = ", ".join(kernel_args)

        step_args = list(spec.step_args)
        step_signature = ", ".join(step_args)

        result_names = self._generate_result_names(spec)
        result_signature = ", ".join(result_names)
        step_call = (
            f"{result_signature} = "
            f"{spec.step_name}({step_signature})"
        )
        observers = self._generate_observers(spec)
        output = self._generate_vectorized_output(spec)

        function_body = self._join_blocks(step_call, observers, output)
        source = (
            f"def {spec.name}({kernel_signature}):\n"
            f"{textwrap.indent(function_body, '    ')}"
        )

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
            step_source=self._generate_step_source(spec),
        )

    def _generate_vectorized_output(self, spec):
        values = ["rhs_value"]

        for name in spec.state_vars:
            if name != "u":
                values.append(f"{name}_new")

        return f"return {', '.join(values)}"
