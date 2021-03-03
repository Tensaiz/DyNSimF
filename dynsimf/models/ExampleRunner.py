from dynsimf.examples.craving_vs_self_control import CravingSelfControl
from dynsimf.examples.HIOM import HIOM

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class ExampleRunner(object):
    '''
    This class contains the references to all the created examples that can be simulated.

    They can then be ran and visualized using this class

    :var str example: The name of the example to simulate
    :var object model: An object of the example class that should be simulated
    '''
    def __init__(self, example):
        self.example = example
        self.initialize_model()

    def initialize_model(self):
        '''
        Initialize the correct model to simulate by creating an object of the right example class
        '''
        if self.example == 'HIOM':
            self.model = HIOM()
        elif self.example == 'Craving vs Self control':
            self.model = CravingSelfControl()

    def simulate(self, n):
        '''
        Simulate the model for `n` iterations and return the results

        :param int n: The amount of iterations to simulate the model
        :return: The output of the simulation
        :rtype: dict
        '''
        return self.model.simulate(n)

    def visualize(self, output):
        '''
        Visualize the output of the model simulation

        :param dict output: The output of the simulation to visualize
        '''
        self.model.visualize(output)
