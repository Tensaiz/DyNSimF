__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"

class PropertyFunction(object):

    def __init__(self, name, function, iteration_interval, params):
        self.name = name
        self.fun = function
        self.iteration_interval = iteration_interval
        self.params = params

    def execute(self):
        return self.fun(**self.params)
