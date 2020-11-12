from dynsimf.models.Model import Model
from dynsimf.models.tools.SA import SensitivityAnalysis
from dynsimf.models.tools.SA import SAConfiguration

import networkx as nx
from networkx.algorithms import average_clustering
import numpy as np

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


if __name__ == "__main__":
    g = nx.random_geometric_graph(200, 0.125)
    model = Model(g)

    constants = {
        'q': 0.8,
        'b': 0.5,
        'd': 0.2,
        'h': 0.2,
        'k': 0.25,
        'S+': 0.5,
    }
    constants['p'] = 2*constants['d']

    def initial_v(constants):
        return np.minimum(1, np.maximum(0, model.get_state('C') - model.get_state('S') - model.get_state('E')))

    def initial_a(constants):
        return constants['q'] * model.get_state('V') + (np.random.poisson(model.get_state('lambda'))/7)

    initial_state = {
        'C': 0,
        'S': constants['S+'],
        'E': 1,
        'V': initial_v,
        'lambda': 0.5,
        'A': initial_a
    }

    def update_C(constants):
        c = model.get_state('C') + constants['b'] * model.get_state('A') * np.minimum(1, 1-model.get_state('C')) - constants['d'] * model.get_state('C')
        return {'C': c}

    def update_S(constants):
        return {'S': model.get_state('S') + constants['p'] * np.maximum(0, constants['S+'] - model.get_state('S')) - constants['h'] * model.get_state('C') - constants['k'] * model.get_state('A')}

    def update_E(constants):
        # return {'E': model.get_state('E') - 0.015}
        e = np.zeros(len(model.nodes))
        for i, node in enumerate(model.nodes):
            neighbor_addiction = 0
            for neighbor in model.get_neighbors(node):
                neighbor_addiction += model.get_node_state(neighbor, 'A')
            e[i] = neighbor_addiction / 50
        return {'E': np.maximum(-1.5, model.get_state('E') - e)} # Custom calculation

    def update_V(constants):
        return {'V': np.minimum(1, np.maximum(0, model.get_state('C')-model.get_state('S')-model.get_state('E')))}

    def update_lambda(constants):
        return {'lambda': model.get_state('lambda') + 0.01}

    def update_A(constants):
        return {'A': constants['q'] * model.get_state('V') + np.minimum((np.random.poisson(model.get_state('lambda'))/7), constants['q']*(1 - model.get_state('V')))}

    # Model definition
    model.constants = constants
    model.set_states(['C', 'S', 'E', 'V', 'lambda', 'A'])
    model.add_update(update_C, {'constants': model.constants})
    model.add_update(update_S, {'constants': model.constants})
    model.add_update(update_E, {'constants': model.constants})
    model.add_update(update_V, {'constants': model.constants})
    model.add_update(update_lambda, {'constants': model.constants})
    model.add_update(update_A, {'constants': model.constants})
    # model.set_initial_state(initial_state, {'constants': model.constants})

    cfg = SAConfiguration(
        {
            'bounds': {'q': (0.79, 0.81), 'b': (0.49, 0.51), 'd': (0.19, 0.21)},
            'iterations': 100,
            'initial_state': initial_state,
            'initial_args': {'constants': model.constants},
            'n': 2,
            'second_order': True,

            'algorithm_input': 'network',
            'algorithm': average_clustering,
            'output_type': 'reduce',
            'algorithm_args': {},
        }
    )
    sa = SensitivityAnalysis(cfg, model)
    analysis = sa.analyze_sensitivity()
    print(analysis)
