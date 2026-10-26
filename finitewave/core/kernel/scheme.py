from abc import ABC, abstractmethod


class KernelScheme(ABC):
    """Generates a model specification into backend-neutral Python source."""

    @abstractmethod
    def generate(self, spec):
        raise NotImplementedError


    # Common methods for kernel generation that can be used by subclasses:
    
    def _collect_kernel_arguments(self, spec):
        result = []

        for name in (
            *spec.common_args,
            *spec.arrays,
            *spec.scalars,
        ):
            if name not in result:
                result.append(name)

        for observer in spec.observers:
            for name in observer.kernel_args:
                if name not in result:
                    result.append(name)

        return result

    def _generate_step_arguments(self, spec):
        args = []

        array_names = set(spec.arrays)
        array_names.add("u")

        for name in spec.step_args:
            if name in array_names:
                args.append(f"{name}.flat[index]")
            else:
                args.append(name)

        return args

    def _generate_observers(self, spec):
        lines = [
            f"{observer.name}_[index] = {observer.expression}"
            for observer in spec.observers
        ]

        return self._indent(lines)

    def _generate_output(self, spec):
        values = ", ".join(f"{name}_" for name in spec.output_args)
        return f"return {values}"

    @staticmethod
    def _indent(lines):
        if not lines:
            return "pass"

        return "\n".join(lines)




def remove_decorators(source):
    """Remove decorators - compilation belongs to backend contexts."""
    lines = source.splitlines()
    while lines and lines[0].lstrip().startswith("@"):
        lines.pop(0)
    return "\n".join(lines)
