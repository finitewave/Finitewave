from abc import ABC, abstractmethod


class SolverBase(ABC):
    """
    A base class for solvers used to solve linear systems.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def initialize(self, simulation):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError