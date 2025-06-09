

class IncorrectWeightsShapeError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class IncorrectNumberOfWeights(Exception):
    """Exception raised for errors in the shape of weights in the
    ``CardiacTissue`` class.

    This exception is used to indicate that the shape of weights provided does
    not match the expected dimensions. It includes details about the incorrect
    shape and the expected shapes.

    Parameters
    ----------
    number_of_weights : int
        The number of weights in the incorrect shape.

    n1 : int
        The target number of weights in one dimension.

    n2 : int
        The target number of weights in another dimension.
    """

    def __init__(self, number_of_weights, n1, n2):
        """
        Initializes the ``IncorrectNumberOfWeights`` with details about the
        incorrect shape and expected dimensions.

        Parameters
        ----------
        number_of_weights : int
            The number of weights in the incorrect shape.

        n1 : int
            The target number of weights in one dimension.

        n2 : int
            The target number of weights in another dimension.
        """
        self.message = (f"Number of weights provided ({number_of_weights})" +
                        f"does not match the expected {n1} or {n2}.")
        super().__init__(self.message)
