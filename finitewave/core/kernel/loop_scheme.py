import textwrap

from .artifact import KernelArtifact
from .scheme import KernelScheme


class LoopKernelScheme(KernelScheme):
    """Generate a scalar kernel with an explicit loop over tissue indexes."""

    def generate(self, spec):
        kernel_args = self._collect_kernel_arguments(spec)
        kernel_signature = ", ".join(kernel_args)

        step_args = self._generate_step_arguments(spec)
        step_signature = ", ".join(step_args)

        result_names = self._generate_result_names(spec)
        result_signature = ", ".join(result_names)

        step_call = (
            f"{result_signature} = "
            f"{spec.step_name}({step_signature})"
        )
        updates = self._generate_state_updates(spec)
        observers = self._generate_observers(spec)
        output = self._generate_output(spec)

        loop_body = self._join_blocks(step_call, updates, observers)
        loop = (
            "for position in kernel_range(len(indexes)):\n"
            "    index = indexes[position]\n"
            f"{textwrap.indent(loop_body, '    ')}"
        )
        function_body = self._join_blocks(loop, output)
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

    def _generate_state_updates(self, spec):
        lines = ["rhs[index] = rhs_value"]

        for name in spec.state_vars:
            if name == "u":
                continue

            lines.append(f"{name}[index] = {name}_new")

        return self._join_lines(lines)
