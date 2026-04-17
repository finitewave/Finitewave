from abc import ABC, abstractmethod


class Solver(ABC):
    """
    A base class for solvers used to solve linear systems.
    """
    def __init__(self):
        pass

    @abstractmethod
    def assemble_system(self, stiffness_matrix, mass_matrix, dt):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError
