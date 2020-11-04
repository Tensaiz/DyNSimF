import numpy as np

from dynsimf.models.conditions.Condition import Condition
from dynsimf.models.conditions.Condition import ConditionType

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class StochasticCondition(Condition):
    def __init__(self, condition_type, p, chained_condition=None):
        super(StochasticCondition, self).__init__(condition_type, chained_condition)
        self.probability = p
        self.validate()

    def validate(self):
        """
        Validate whether the state and threshold are in correct formats
        """
        if not (isinstance(self.probability, int) or isinstance(self.probability, float)):
            raise ValueError('Probability should be an integer or float')

    def get_arguments(self, model_input):
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
        return self.random_sample

    def random_sample(self, nodes):
        sampled_probabilities = np.random.random_sample(len(nodes))
        return nodes[np.where(sampled_probabilities < self.probability)[0]]

    def test_states(self, nodes):
        pass

    def test_utility(self, nodes):
        pass

    def test_adjacency(self, nodes):
        pass
