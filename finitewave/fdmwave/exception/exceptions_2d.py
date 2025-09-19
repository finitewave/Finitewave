class IncorrectWeightsModeError2D(Exception):
    """
    Exception raised for invalid modes in the CardiacTissue2D class.

    Attributes
    ----------
    mode : str
        The invalid mode that caused the exception.
    message : str
        Explanation of the error.
    """

    def __init__(self, mode, message="CardiacTissue2D mode attribute must be 'iso' or 'aniso'"):
        """
        Initializes the IncorrectWeightsModeError2D exception.

        Parameters
        ----------
        mode : str
            The invalid mode that caused the exception.
        message : str, optional
            Explanation of the error (default is "CardiacTissue2D mode attribute must be 'iso' or 'aniso'").
        """
        self.mode = mode
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        """
        Returns a string representation of the exception.

        Returns
        -------
        str
            A string describing the error including the invalid mode.
        """
        return f"{self.message} (Invalid mode: '{self.mode}')"
