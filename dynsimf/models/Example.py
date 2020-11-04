from abc import ABCMeta, abstractclassmethod

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class Example(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractclassmethod
    def simulate(self):
        pass

    @abstractclassmethod
    def visualize(self, iterations):
        pass
