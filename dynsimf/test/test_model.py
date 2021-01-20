import unittest

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration

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


    def test_assign_constant(self):
        # Network definition
        g = nx.random_geometric_graph(10, 0.1)
        m = Model(g)

        initial_state = {
            'x': 0,
        }

        def update_x():
            return {'x': 1}

        # Model definition
        m.set_states(['x'])
        m.add_update(update_x)
        m.set_initial_state(initial_state)

        output = m.simulate(1)
        self.assertEqual(list(output['states'][0]), list(np.zeros((10,1))))
        self.assertEqual(list(output['states'][1]), list(np.ones((10,1))))

    def test_assign_array(self):
        # Network definition
        g = nx.random_geometric_graph(10, 0.1)
        m = Model(g)

        initial_state = {
            'x': 0,
        }

        def update_x():
            return {'x': np.arange(10)}

        # Model definition
        m.set_states(['x'])
        m.add_update(update_x)
        m.set_initial_state(initial_state)

        output = m.simulate(1)
        self.assertEqual(list(output['states'][0]), list(np.zeros((10,1))))
        self.assertEqual(list(output['states'][1]), list(np.arange(10)))


    def test_assign_nodes(self):
        # Network definition
        g = nx.random_geometric_graph(10, 0.1)
        m = Model(g)

        initial_state = {
            'x': 0,
        }

        def update_x():
            return {'x': {0: 5, 5: 10}}

        # Model definition
        m.set_states(['x'])
        m.add_update(update_x)
        m.set_initial_state(initial_state)

        output = m.simulate(1)
        # self.assertEqual(list(output['states'][0]), list(np.zeros((10,1))))
        # self.assertEqual(list(output['states'][1]), list(np.arange(10)))
        print(output)
