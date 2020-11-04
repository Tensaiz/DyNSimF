import unittest

from dynsimf.models.Model import Model

import networkx as nx
import numpy as np

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class ModelTest(unittest.TestCase):
    def test_model_init(self):
        g = nx.random_geometric_graph(10, 0.1)
        m = Model(g)
        self.assertTrue(isinstance(m, Model))

    def test_model_constants(self):
        g = nx.random_geometric_graph(10, 0.1)
        m = Model(g)
        d = {1: 2}
        m.constants = d
        self.assertEqual(m.constants, d)

    def test_set_states(self):
        g = nx.random_geometric_graph(10, 0.1)
        m = Model(g)
        m.set_states(['1', '2'])
        self.assertTrue((np.zeros((10, 2)) == m.node_states).any())
        self.assertEqual(m.state_names, ['1', '2'])
        self.assertEqual(m.state_map, {'1': 0, '2': 1})
