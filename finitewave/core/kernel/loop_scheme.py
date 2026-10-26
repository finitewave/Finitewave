import textwrap

from .artifact import KernelArtifact
from .scheme import KernelScheme, remove_decorators


class LoopKernelScheme(KernelScheme):
    """Generate a scalar kernel with an explicit loop over tissue indexes."""

    # def __init__(self, loop_function: str = "kernel_range"):
    #     self.loop_function = loop_function

    def generate(self, spec):
        kernel_args = self._collect_kernel_arguments(spec)
        kernel_signature = ", ".join(kernel_args)

        step_args = self._generate_step_arguments(spec)
        step_signature = ", ".join(step_args)

        result_names = ["rhs_new"]
        for name in spec.state_vars:
            if name == "u":
                continue
        result_names.append(f"{name}_new")
        result_signature = ", ".join(result_names)

        updates = ["rhs.flat[index] = rhs_new"]

        updates = self._generate_state_updates(spec)
        observers = self._generate_observers(spec)
        output = self._generate_output(spec)

        source = f"""
        def {spec.name}({kernel_signature}):
            for index in indexes:
                {result_signature} = {spec.step_name}(
                    {step_signature}
                )

        {updates}

            return {output}
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

    def _generate_state_updates(self, spec):
        lines = []

        for name in spec.state_vars:
            if name == "u":
                continue

            lines.append(f"{name}_[index] = {name}_new")

        lines.append("rhs_[index] = rhs")

        return self._indent(lines)
