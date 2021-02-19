from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class ConditionType(Enum):
    '''
    The condition type enum, specifying what the condition is targeting
    '''
    STATE = 0
    ADJACENCY = 1
    UTILITY = 2
    CUSTOM = 3
    EDGE_VALUES = 4

class Condition(metaclass=ABCMeta):
    '''
    Condition base class
    '''
    def __init__(self, condition_type, chained_condition=None):
        '''
        Initialize the condition

        :param ConditionType condition_type: The type of the condition
        :param chained_condition: Any Condition that should also apply for the nodes.
            It will automatically be checked after the current condition.
        :type chained_condition: Condition, optional
        '''
        if not isinstance(condition_type, ConditionType):
            raise ValueError("The provided condition type is not valid")
        self.condition_type = condition_type
        self.chained_condition = chained_condition
        self.validate_condition()
        self.config = None

    def validate_condition(self):
        '''
        Validate whether a condition is configured correctly

        :raises ValueError: if an uncorrect condition_type is specified at initialisation
        :raises ValueError: if the chained condition is not of Condition type
        '''
        if not isinstance(self.condition_type, ConditionType):
            raise ValueError('Condition type should be a ConditionType enumerated value')
        if self.chained_condition and not issubclass(type(self.chained_condition), Condition):
            raise ValueError('A chained condition must be a Condition subclass')

    def get_state(self):
        '''
        Get the state name that the condition is targeting

        :return: The state the condition targets
        :rtype: str or None
        '''
        if self.config and self.config.state:
            return self.config.state
        elif 'state' in self.__dict__:
            return self.state
        else:
            return None

    def get_valid_nodes(self, model_input):
        '''
        Get all valid nodes that meet this and the chained conditions

        :param tuple model_input: Gives all important model values from the current iteration: 
            (nodes, states matrix, adjacency matrix, utility matrix)
        :return: List of all nodes that meet the condition
        :rtype: list
        '''
        _, states, adjacency_matrix, utility_matrix = model_input
        f = self.get_function()
        args = self.get_arguments(model_input)

        selected_nodes = f(*args)

        return selected_nodes \
            if not self.chained_condition \
            else self.chained_condition.get_valid_nodes((selected_nodes, states, adjacency_matrix, utility_matrix))

    def get_function(self):
        '''
        Get the function that takes the model input and returns the valid nodes that meet the condition.
        Note that this function is overwritten when using the CustomCondition class

        :return: The function that returns the nodes for which the condition applies
        :rtype: function
        '''
        condition_type_to_function_map = {
            ConditionType.STATE: self.test_states,
            ConditionType.UTILITY: self.test_utility,
            ConditionType.ADJACENCY: self.test_adjacency
        }
        return condition_type_to_function_map[self.condition_type]

    def test_adjacency(self):
        '''
        Empty function that should be specified in a subclass to test for an adjacency condition
        '''
        pass

    def test_utility(self):
        '''
        Empty function that should be specified in a subclass to test for a utility condition
        '''
        pass

    def test_states(self):
        '''
        Empty function that should be specified in a subclass to test for a state condition
        '''
        pass

    @abstractmethod
    def get_arguments(self, model_input):
        '''
        Empty function that should be specified in a subclass 
        to get the desired arguments for the test function. 
        Note that this function must take in the `model_input` tuple of the form: 
        (nodes, states matrix, adjacency matrix, utility matrix)

        :param tuple model_input: Standard parameter for any condition sample function, 
            it has the form: (nodes, states matrix, adjacency matrix, utility matrix)
        '''
        pass
