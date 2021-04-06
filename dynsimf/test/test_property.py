import unittest

import numpy as np
import random
import networkx as nx

from dynsimf.models.Model import Model
from dynsimf.models.components.PropertyFunction import PropertyFunction


__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class PropertyTest(unittest.TestCase):
    def test_properties(self):
        np.random.seed(1337)
        random.seed(1337)

        g = nx.random_geometric_graph(50, 0.3)
        model = Model(g)

        def node_amount(G):
            return len(G.nodes())

        prop1 = PropertyFunction('1', nx.average_clustering, 1, {'G': model.graph})
        prop2 = PropertyFunction('2', node_amount, 1, {'G': model.graph})

        model.add_property_function(prop1)
        model.add_property_function(prop2)

        model.simulate(3)
        out = model.get_properties()
        
        self.assertDictEqual(out, {'1': [0.6747703316560499, 0.6747703316560499, 0.6747703316560499], '2': [50, 50, 50]})
