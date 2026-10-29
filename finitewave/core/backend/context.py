from abc import ABC, abstractmethod


class BackendContext(ABC):

    # Namespace examples:
    # Numba: 
    # { 
    # "kernel_range": numba.prange,
    # "exp": numpy.exp,
    # "log": numpy.log,
    # "sqrt": numpy.sqrt,
    # "calc_rhs": compiled_calc_rhs,
    # ...
    # }
    # Jax (mlx is the same):
    # { 
    # "kernel_range": range,
    # "exp": jax.numpy.exp,
    # "log": jax.numpy.log,
    # "sqrt": jax.numpy.sqrt,
    # "calc_where": jax.numpy.where,
    # "calc_rhs": jax_calc_rhs,
    # ...
    # }

    name = ""

    def build(self, artifact):
        """
        Build Python functions from step_source and kernel source.
        """
        namespace = self.kernel_namespace(
            artifact.spec
        )

        artifact.python_step = self._build_function(
            name=artifact.step_name,
            source=artifact.step_source,
            namespace=namespace,
        )

        artifact.compiled_step = self.compile_step(
            artifact.python_step,
            spec=artifact.spec,
        )

        kernel_namespace = dict(namespace)
        kernel_namespace[artifact.step_name] = (
            artifact.compiled_step
        )

        artifact.python_function = self._build_function(
            name=artifact.spec.name,
            source=artifact.source,
            namespace=kernel_namespace,
        )

        return artifact.python_function

    def compile(self, artifact):
        """
        Compile the complete spatial kernel.
        """
        if artifact.python_function is None:
            self.build(artifact)

        artifact.compiled_function = self.compile_kernel(
            artifact.python_function,
            spec=artifact.spec,
        )

        return artifact.compiled_function

    @staticmethod
    def _build_function(name, source, namespace):
        local_namespace = {}

        exec(
            source,
            namespace,
            local_namespace,
        )

        return local_namespace[name]

    @abstractmethod
    def kernel_namespace(self, spec):
        """
        Globals used by generated step and kernel functions.
        """
        raise NotImplementedError

    def compile_step(self, function, *, spec):
        """
        Compile the cell-level ionic step.

        Default implementation leaves it as a Python function.
        """
        return function

    @abstractmethod
    def compile_kernel(self, function, *, spec):
        """
        Compile the complete spatial kernel.
        """
        raise NotImplementedError

    def synchronize(self, *values):
        return values