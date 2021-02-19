import inspect

from dynsimf.models.components.conditions.Condition import Condition
from dynsimf.models.components.conditions.Condition import ConditionType

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"

class CustomCondition(Condition):
    '''
    Base class that can be used to create custom conditions that do not fit under stochastic or threshold conditions
    '''
    def __init__(self, function, arguments=None, chained_condition=None):
        '''
        Initialise the custom condition by setting the function and arguments, then validating itself

        :param function function: The function that tests and returns the nodes that meet the condition.
            Note that this function must take in an argument called 'model_input' of the form:
            (nodes, states matrix, adjacency matrix, utility matrix).
        :param arguments: Extra arguments that the function should receive apart from the standard model_input tuple
        :type arguments: any, optional
        :param chained_condition: Any Condition that should also apply for the nodes.
            It will automatically be checked after the current condition.
        :type chained_condition: Condition, optional
        '''
        super(CustomCondition, self).__init__(ConditionType.CUSTOM, chained_condition)
        self.sample_function = function
        self.arguments = [] if not arguments else arguments
        self.validate()

    def validate(self):
        '''
        Validate the custom condition

        :raises ValueError: if no function is provided
        :raises ValueError: if the function does not take the 'model_input' tuple as argument 
        '''
        if not callable(self.sample_function):
            raise ValueError('The function argument should contain a function')
        if 'model_input' not in inspect.getfullargspec(self.sample_function).args:
            raise ValueError('The custom condition function should take a model_input argument, that is provided when the function is called')

    def get_function(self):
        '''
        Get the sample function of the custom condition

        :return: The sample function of the custom condition
        :rtype: function
        '''
        return self.sample_function

    def get_arguments(self, model_input):
        '''
        Get the arguments for the custom condition, note that this returns the model_input tuple 
        plus whatever custom arguments have been specified

        :param tuple model_input: the model_input tuple of form:
            (nodes, states matrix, adjacency matrix, utility matrix) 
            that is automatically provided to the sample function of the custom condition
        :return: A list of arguments, where the first argument is the model_input tuple 
            and the other arguments are any custom provided arguments
        :rtype: list
        '''
        return [model_input] + self.arguments
