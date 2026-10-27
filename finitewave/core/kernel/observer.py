

class ObserverSpec:

    # example:
    # name="ina_values"
    # expression="ina_values[index] = ina"
    # step_outputs=("ina")
    # kernel_args=("ina_values")

    def __init__(
        self,
        name,
        expression,
        step_outputs,
        kernel_args,
    ):
        self.name = name
        self.expression = expression
        self.step_outputs = tuple(step_outputs)
        self.kernel_args = tuple(kernel_args)