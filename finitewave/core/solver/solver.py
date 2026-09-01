from abc import ABC, abstractmethod


class Solver(ABC):
    """
    A base class for integration solvers.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def initialize(self, simulation):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError