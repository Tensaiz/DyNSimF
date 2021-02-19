__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"

class PropertyFunction(object):
    '''
    Class used to specify property functions, 
    that are meta analysis functions that can be ran during any iteration 
    to run custom values.

    E.g: Calculate the network cluster coeffecient every 5 iterations 
    '''
    def __init__(self, name, function, iteration_interval, params):
        '''
        Initialise the property function 

        :param str name: The name of the property function
        :param function function: the function to run
        :param int iteration_interval: The interval between executions of the function
        :param dict params: A dictionary containing the arguments to provide for the function
        '''
        self.name = name
        self.fun = function
        self.iteration_interval = iteration_interval
        self.params = params

    def execute(self):
        '''
        Executes the function with the specified arguments and returns the result

        :return: The result of the property function
        '''
        return self.fun(**self.params)
