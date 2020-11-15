import operator
import warnings
from enum import Enum

import numpy as np

from dynsimf.models.components.conditions.Condition import Condition
from dynsimf.models.components.conditions.Condition import ConditionType

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class ThresholdOperator(Enum):
    GT = operator.__gt__
    LT = operator.__lt__
    GE = operator.__ge__
    LE = operator.__le__


class ThresholdConfiguration(object):
    def __init__(self, threshold_operator, threshold, state=None):
        self.threshold_operator = threshold_operator
        self.threshold = threshold
        self.state = state
        self.state_index = None
        self.validate()

    def validate(self):
        """
        Validate whether the state and threshold are in correct formats
        """
        if not isinstance(self.threshold_operator, ThresholdOperator):
            raise ValueError('Invalid threshold type')


class ThresholdCondition(Condition):
    def __init__(self, condition_type, config, chained_condition=None):
        super(ThresholdCondition, self).__init__(condition_type, chained_condition)
        self.config = config
        self.validate()

    def validate(self):
        """
        Validate whether the state and threshold are in correct formats
        """
        if not isinstance(self.config, ThresholdConfiguration):
            raise ValueError('Configuration object should be of class ThresholdConfiguration')
        if self.condition_type == ConditionType.STATE and not self.config.state:
            raise ValueError('A state should be provided when using state type')
        if self.condition_type != ConditionType.STATE and self.config.state:
            warnings.warn('The condition type has not been set to state, but a state has been set. The set state will be ignored')

    def set_state_index(self, index):
        self.config.state_index = index

    def get_arguments(self, model_input):
        nodes, states, adjacency_matrix, utility_matrix = model_input
        condition_type_to_arguments_map = {
            ConditionType.STATE: [
                nodes,
                states
            ],
            ConditionType.UTILITY: [
                nodes,
                utility_matrix
            ],
            ConditionType.ADJACENCY: [
                nodes,
                adjacency_matrix
            ]
        }
        return condition_type_to_arguments_map[self.condition_type]

    def test_states(self, nodes, states):
        """
        Find the indices of nodes for which the threshold operator holds for the set state index
        """
        if not self.config.state_index:
            raise ValueError('State index has not been set')
        return nodes[np.where(self.config.threshold_operator.value(states[nodes, self.config.state_index], self.config.threshold))[0]]

    def test_utility(self, nodes, utility_matrix):
        """
        Find the indices of nodes that have at least one utility edge value for which the threshold operator holds

        Returns indices of the nodes for which `threshold operator(utility_edge, condition.threshold)` is true
        """
        return np.unique(np.where(self.config.threshold_operator.value(utility_matrix[nodes], self.config.threshold))[0])

    def test_adjacency(self, nodes, adjacency_matrix):
        """
        Calculates the amount of neighbors for each node and applies the threshold operator on each node's neighbor amount

        Returns indices of the nodes for which: `threshold operator(n_neighbors, condition.threshold)` is true
        """
        return nodes[np.where(self.config.threshold_operator.value(np.sum(adjacency_matrix[nodes], 1), self.config.threshold))[0]]
