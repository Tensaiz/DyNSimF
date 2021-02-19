import operator
import warnings
from enum import Enum

import numpy as np

from dynsimf.models.components.conditions.Condition import Condition
from dynsimf.models.components.conditions.Condition import ConditionType

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class ThresholdOperator(Enum):
    '''
    An enumeration that is used to select the specific operator for the threshold condition
    '''
    GT = operator.__gt__
    LT = operator.__lt__
    GE = operator.__ge__
    LE = operator.__le__


class ThresholdConfiguration(object):
    '''
    Configuration class used to configure a threshold condition
    '''
    def __init__(self, threshold_operator, threshold, state=None):
        '''
        Initialise the threshold configuration by setting the members and then validating itself

        :param ThresholdOperator threshold_operator: the operator enum value 
            that indicates the type of operator used for the condition
        :param threshold: The threshold value
        :type threshold: int or float
        :param state: The name of the state that the threshold should be checked for (if applicable)
        :type state: str, optional
        '''
        self.threshold_operator = threshold_operator
        self.threshold = threshold
        self.state = state
        self.state_index = None
        self.validate()

    def validate(self):
        '''
        Validate whether the threshold operator is of enum type ThresholdOperator
        '''
        if not isinstance(self.threshold_operator, ThresholdOperator):
            raise ValueError('Invalid threshold type')


class ThresholdCondition(Condition):
    def __init__(self, condition_type, config, chained_condition=None):
        '''
        :param ConditionType condition_type: The type of the condition
        :param ThresholdConfiguration config: The config object for the threshold condition
        :param chained_condition: Any Condition that should also apply for the nodes.
            It will automatically be checked after the current condition.
        :type chained_condition: Condition, optional
        '''
        super(ThresholdCondition, self).__init__(condition_type, chained_condition)
        self.config = config
        self.validate()

    def validate(self):
        '''
        Validate whether the configuration type is correct and if states have been set if the threshold should apply on a state

        :raises ValueError: if the `config` member is not of type `ThresholdConfiguration`
        :raises ValueError: if no state has been set in the config but the `condition_type` member is set to `ConditionType.STATE`
        :raises Warning: if a state has been set in the config, but the `condition_type` member is not `ConditionType.STATE`
        '''
        if not isinstance(self.config, ThresholdConfiguration):
            raise ValueError('Configuration object should be of class ThresholdConfiguration')
        if self.condition_type == ConditionType.STATE and not self.config.state:
            raise ValueError('A state should be provided when using state type')
        if self.condition_type != ConditionType.STATE and self.config.state:
            warnings.warn('The condition type has not been set to state, but a state has been set. The set state will be ignored')

    def set_state_index(self, index):
        '''
        Set the state index corresponding to the selected state in the config. 
        Essentially the state string is now also stored 
        as the specific index that refers to the state in the complete states matrix in the model. 

        :param int index: The index of the `state` in the config that indexes to the state in the model's states matrix
        '''
        self.config.state_index = index

    def get_arguments(self, model_input):
        '''
        Get the arguments for the threshold condition sample function

        :param tuple model_input: the model_input tuple of form:
            (nodes, states matrix, adjacency matrix, utility matrix) 
            that is automatically provided to the sample function of the custom condition
        :return: A list of arguments depending on the specified `condition_type`
        :rtype: list
        '''
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
        '''
        Find the indices of nodes for which the threshold operator holds for the set state index

        :param list nodes: A list of nodes to be sampled
        :param numpy.ndarray states: A matrix of the states in the model at the current iteration
        :return: The nodes that meet the threshold condition
        :rtype: list
        :raises ValueError: if no state index has been set in the `config` member
        '''
        if not self.config.state_index:
            raise ValueError('State index has not been set')
        return nodes[np.where(self.config.threshold_operator.value(states[nodes, self.config.state_index], self.config.threshold))[0]]

    def test_utility(self, nodes, utility_matrix):
        '''
        Find the indices of nodes that have at least one utility edge value for which the threshold operator holds. 
        Returns indices of the nodes for which `threshold operator(utility_edge, condition.threshold)` is true

        :param list nodes: A list of nodes to be sampled
        :param numpy.ndarray utility_matrix: A matrix of the utility in the model at the current iteration
        :return: The nodes that meet the threshold condition
        :rtype: list
        '''
        return np.unique(np.where(self.config.threshold_operator.value(utility_matrix[nodes], self.config.threshold))[0])

    def test_adjacency(self, nodes, adjacency_matrix):
        """
        Calculates the amount of neighbors for each node and applies the threshold operator on each node's neighbor amount

        Returns the nodes for which: `threshold operator(n_neighbors, condition.threshold)` is true

        :param list nodes: A list of nodes to be sampled
        :param numpy.ndarray adjacency_matrix: A matrix of the adjacency in the model at the current iteration
        :return: The nodes that meet the threshold condition
        :rtype: list
        """
        return nodes[np.where(self.config.threshold_operator.value(np.sum(adjacency_matrix[nodes], 1), self.config.threshold))[0]]
