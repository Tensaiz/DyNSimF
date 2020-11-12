import unittest

import numpy as np

from dynsimf.models.components.conditions.StochasticCondition import StochasticCondition
from dynsimf.models.components.conditions.ThresholdCondition import ThresholdCondition
from dynsimf.models.components.conditions.ThresholdCondition import ThresholdOperator
from dynsimf.models.components.conditions.ThresholdCondition import ThresholdConfiguration
from dynsimf.models.components.conditions.Condition import ConditionType
from dynsimf.models.components.conditions.CustomCondition import CustomCondition

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class ConditionsTest(unittest.TestCase):
    def test_threshold_state_condition(self):
        states = np.array([[1, 2, 3], [4, 5, 6], [1, 4, 8]])

        t_cfg = ThresholdConfiguration(ThresholdOperator.GE, 4, 'A')
        t = ThresholdCondition(ConditionType.STATE, t_cfg)
        t.set_state_index(1)
        nodes = t.get_valid_nodes((np.arange(3), states, None, None))

        self.assertEqual(list(nodes), [1, 2])

    def test_threshold_adjacency_condition(self):
        np.random.seed(1337)
        adjacency = np.random.randint(2, size=25).reshape(5, 5)

        t_cfg = ThresholdConfiguration(ThresholdOperator.GE, 3)
        t = ThresholdCondition(ConditionType.ADJACENCY, t_cfg)

        nodes = t.get_valid_nodes((np.arange(5), None, adjacency, None))
        self.assertEqual(list(nodes), [0, 3])

    def test_threshold_utility_condition(self):
        np.random.seed(1337)
        utility = np.random.random_sample(25).reshape(5, 5)

        t_cfg = ThresholdConfiguration(ThresholdOperator.GE, 0.8)
        t = ThresholdCondition(ConditionType.UTILITY, t_cfg)

        nodes = t.get_valid_nodes((np.arange(5), None, None, utility))
        self.assertEqual(list(nodes), [1, 2])

    def test_stochastic_condition_state(self):
        np.random.seed(1337)
        states = np.ones((20, 3))

        s = StochasticCondition(ConditionType.STATE, 0.25)
        nodes = s.get_valid_nodes((np.arange(len(states)), states, None, None))

        self.assertEqual(list(nodes), [1, 9, 12])

    def test_stochastic_condition_utility(self):
        np.random.seed(1337)
        states = np.ones((20, 3))

        s = StochasticCondition(ConditionType.UTILITY, 0.25)
        nodes = s.get_valid_nodes((np.arange(len(states)), states, None, None))

        self.assertEqual(list(nodes), [1, 9, 12])

    def test_chained_conditions(self):
        np.random.seed(1337)
        states = np.random.random_sample((100, 3))

        t_cfg = ThresholdConfiguration(ThresholdOperator.GE, 0.5, 'A')
        t = ThresholdCondition(ConditionType.STATE, t_cfg)
        t.set_state_index(1)

        c = StochasticCondition(ConditionType.STATE, 0.25, chained_condition=t)
        nodes = c.get_valid_nodes((np.arange(len(states)), states, None, None))

        self.assertEqual(list(nodes), [ 6, 10, 12, 22, 39, 42, 48, 62, 79, 90])

    def test_state_adjacency_threshold_conditions(self):
        np.random.seed(1337)
        states = np.random.random_sample((5, 3))
        adjacency = np.random.randint(2, size=25).reshape(5, 5)

        t_cfg_adj = ThresholdConfiguration(ThresholdOperator.GE, 2)
        t_adj = ThresholdCondition(ConditionType.ADJACENCY, t_cfg_adj)

        t_cfg = ThresholdConfiguration(ThresholdOperator.GT, 0.5, 'A')
        t = ThresholdCondition(ConditionType.STATE, t_cfg, chained_condition=t_adj)
        t.set_state_index(1)

        nodes = t.get_valid_nodes((np.arange(len(states)), states, adjacency, None))

        self.assertEqual(list(nodes), [2])

    def test_custom_condition(self):
        np.random.seed(1337)
        states = np.random.random_sample((100, 1))

        def sample(model_input):
            nodes = model_input[0]
            state2 = model_input[1][:, 0]

            chances = (np.ones(len(nodes)) - state2) / 10
            draws = np.random.random_sample(len(nodes))

            indices = np.where(draws < chances)[0]

            return nodes[indices]

        c = CustomCondition(sample)
        nodes = c.get_valid_nodes((np.arange(len(states)), states, None, None))
        self.assertEqual(list(nodes),[ 1, 21, 35, 38, 51, 65, 74, 86])
