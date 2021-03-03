from abc import ABCMeta, abstractclassmethod

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class Example(object, metaclass=ABCMeta):
    '''
    Base meta class for creating examples

    Every example should at least contain a simulate and visualize function
    '''
    def __init__(self):
        pass

    @abstractclassmethod
    def simulate(self):
        '''
        Meta function

        The simulate function should simulate the example
        '''
        pass

    @abstractclassmethod
    def visualize(self, output):
        '''
        Meta function

        The visualization function should take in the output of the simulation and visualize it

        :param dict output: The output from the simulation
        '''
        pass
