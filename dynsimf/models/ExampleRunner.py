from dynsimf.examples.craving_vs_self_control import CravingSelfControl
from dynsimf.examples.HIOM import HIOM

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class ExampleRunner(object):
    def __init__(self, example):
        self.example = example
        self.initialize_model()

    def initialize_model(self):
        if self.example == 'HIOM':
            self.model = HIOM()
        elif self.example == 'Craving vs Self control':
            self.model = CravingSelfControl()

    def simulate(self, n):
        return self.model.simulate(n)

    def visualize(self, output):
        self.model.visualize(output)
