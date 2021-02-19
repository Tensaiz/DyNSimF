import numpy as np

from dynsimf.models.components.conditions.Condition import Condition
from dynsimf.models.components.conditions.Condition import ConditionType

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class StochasticCondition(Condition):
    '''
    A class for creating a stochastic condition where any node has a certain chance of being randomly sampled
    '''
    def __init__(self, condition_type, p, chained_condition=None):
        '''
        Initialize the stochastic condition, the sample probability is set and the condition is validated

        :param ConditionType condition_type: The type of the condition
        :param p: The probability of a node being sampled
        :type p: float or int
        :param chained_condition: Any Condition that should also apply for the nodes.
            It will automatically be checked after the current condition.
        :type chained_condition: Condition, optional
        '''
        super(StochasticCondition, self).__init__(condition_type, chained_condition)
        self.probability = p
        self.validate()

    def validate(self):
        '''
        Validate whether the probability is in the correct format

        :raises ValueError: if the probability member is neither a float or int 
        '''
        if not (isinstance(self.probability, int) or isinstance(self.probability, float)):
            raise ValueError('Probability should be an integer or float')

    def get_arguments(self, model_input):
        '''
        Get the arguments for the stochastic condition sample function.
        In this case, no matter the condition type, only the nodes are returned from the model_input

        :param tuple model_input: the model_input tuple of form:
            (nodes, states matrix, adjacency matrix, utility matrix)
        :return: A list of all nodes in the model
        :rtype: list
        '''
        nodes, _, _, _ = model_input
        condition_type_to_arguments_map = {
            ConditionType.STATE: [
                nodes
            ],
            ConditionType.UTILITY: [
                nodes
            ],
            ConditionType.ADJACENCY: [
                nodes
            ]
        }
        return condition_type_to_arguments_map[self.condition_type]

    def get_function(self):
        '''
        Get the sample function of the stochastic condition

        :return: The sample function
        :rtype: function
        '''
        return self.random_sample

    def random_sample(self, nodes):
        '''
        The function that samples the nodes in the model based on the configured probability.
        For every node a random number is generated 
        and if one of the drawn numbers is less than the selected probability 
        that node is selected and returned together with the other selected nodes.

        :return: A list of sampled nodes
        :rtype: list
        '''
        sampled_probabilities = np.random.random_sample(len(nodes))
        return nodes[np.where(sampled_probabilities < self.probability)[0]]

    def test_states(self, nodes):
        '''
        Empty function that is not used because the `random_sample` function is used instead
        '''
        pass

    def test_utility(self, nodes):
        '''
        Empty function that is not used because the `random_sample` function is used instead
        '''
        pass

    def test_adjacency(self, nodes):
        '''
        Empty function that is not used because the `random_sample` function is used instead
        '''
        pass
