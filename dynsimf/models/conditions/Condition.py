from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class ConditionType(Enum):
    STATE = 0
    ADJACENCY = 1
    UTILITY = 2
    CUSTOM = 3


class Condition(metaclass=ABCMeta):
    """
    Condition base class
    """
    def __init__(self, condition_type, chained_condition=None):
        if not isinstance(condition_type, ConditionType):
            raise ValueError("The provided condition type is not valid")
        self.condition_type = condition_type
        self.chained_condition = chained_condition
        self.validate_condition()
        self.config = None

    def validate_condition(self):
        if not isinstance(self.condition_type, ConditionType):
            raise ValueError('Condition type should be a ConditionType enumerated value')
        if self.chained_condition and not issubclass(type(self.chained_condition), Condition):
            raise ValueError('A chained condition must be a Condition subclass')

    def get_state(self):
        if self.config and self.config.state:
            return self.config.state
        elif 'state' in self.__dict__:
            return self.state
        else:
            return None

    def get_valid_nodes(self, model_input):
        _, states, adjacency_matrix, utility_matrix = model_input
        f = self.get_function()
        args = self.get_arguments(model_input)

        selected_nodes = f(*args)

        return selected_nodes \
            if not self.chained_condition \
            else self.chained_condition.get_valid_nodes((selected_nodes, states, adjacency_matrix, utility_matrix))

    def get_function(self):
        condition_type_to_function_map = {
            ConditionType.STATE: self.test_states,
            ConditionType.UTILITY: self.test_utility,
            ConditionType.ADJACENCY: self.test_adjacency
        }
        return condition_type_to_function_map[self.condition_type]

    def test_adjacency(self):
        pass

    def test_utility(self):
        pass

    def test_states(self):
        pass

    @abstractmethod
    def get_arguments(self, model_input):
        pass
