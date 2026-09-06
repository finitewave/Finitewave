from abc import ABC, abstractmethod



class TimeIntegration(ABC):
    """
    A base class for time integration methods in cardiac simulations.
    This class defines the interface and common functionality for
    different time integration schemes.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def initialize(self, simulation):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError
