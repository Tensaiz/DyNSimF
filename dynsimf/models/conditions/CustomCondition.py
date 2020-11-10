import inspect

from dynsimf.models.conditions.Condition import Condition
from dynsimf.models.conditions.Condition import ConditionType

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"

class CustomCondition(Condition):
    def __init__(self, function, arguments=None, chained_condition=None):
        super(CustomCondition, self).__init__(ConditionType.CUSTOM, chained_condition)
        self.sample_function = function
        self.arguments = [] if not arguments else arguments
        self.validate()

    def validate(self):
        if not callable(self.sample_function):
            raise ValueError('The function argument should contain a function')
        if 'model_input' not in inspect.getfullargspec(self.sample_function).args:
            raise ValueError('The custom condition function should take a model_input argument, that is provided when the function is called')

    def get_function(self):
        return self.sample_function

    def get_arguments(self, model_input):
        self.arguments.insert(0, model_input)
        return self.arguments
