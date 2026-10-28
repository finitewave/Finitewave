from abc import ABC, abstractmethod
import ast


class KernelScheme(ABC):
    """Generates a model specification into backend-neutral Python source."""

    @abstractmethod
    def generate(self, spec):
        raise NotImplementedError


    # Common methods for kernel generation that can be used by subclasses:
    
    def _collect_kernel_arguments(self, spec):
        result = []

        for name in (*spec.common_args, *spec.arrays, *spec.scalarsб *spec.observer_kernel_args):
            if name not in result:
                result.append(name)

        return result

    def _generate_step_arguments(self, spec):
        args = []

        array_names = set(spec.arrays)
        array_names.add("u")

        for name in spec.step_args:
            if name in array_names:
                args.append(f"{name}[index]")
            else:
                args.append(name)

        return args

    def _generate_observers(self, spec):
        lines = [observer.expression for observer in spec.observers]

        return self._join_lines(lines)

    def _generate_output(self, spec):
        values = ", ".join(spec.output_args)
        return f"return {values}"

    @staticmethod
    def _join_lines(lines):
        return "\n".join(lines)

    @staticmethod
    def _join_blocks(*blocks):
        return "\n\n".join(
            block for block in blocks if block and block.strip()
        )

    def _generate_result_names(self, spec):
        names = ["rhs_value"]

        for name in spec.state_vars:
            if name != "u":
                names.append(f"{name}_new")

        for name in spec.observer_step_outputs:
            if name not in names:
                names.append(name)

        return names

    def _generate_step_source(self, spec):
        """Add values required by observers to ionic_step's return."""
        if not spec.observer_step_outputs:
            return spec.step_source

        tree = ast.parse(spec.step_source)
        function = tree.body[0]
        returns = [
            node for node in function.body
            if isinstance(node, ast.Return)
        ]

        if len(returns) != 1:
            raise ValueError(
                "ionic_step must contain exactly one top-level return statement."
            )

        return_node = returns[0]
        if isinstance(return_node.value, ast.Tuple):
            values = list(return_node.value.elts)
        else:
            values = [return_node.value]

        returned_names = {
            value.id for value in values if isinstance(value, ast.Name)
        }
        for name in spec.observer_step_outputs:
            if name not in returned_names:
                values.append(ast.Name(id=name, ctx=ast.Load()))

        return_node.value = ast.Tuple(elts=values, ctx=ast.Load())
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

