import networkx as nx
import numpy as np

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration
from dynsimf.models.components.Scheme import Scheme
from dynsimf.models.components.Update import Update
from dynsimf.models.components.Update import UpdateConfiguration
from dynsimf.models.Example import Example

if __name__ == "__main__":

    n = 400

    g = nx.watts_strogatz_graph(n, 2, 0.02)

    cfg = {
        'utility': False,
    }
    model = Model(g, ModelConfiguration(cfg))

    constants = {
        'dt': 0.01,
        'A_min': -0.5,
        'A_star': 1,
        's_O': 0.01,
        's_I': 0,
        'd_A': 0,
        'p': 1,
        'r_min': 0,
        't_O': np.inf,
        'N': n
    }

    def initial_I(constants):
        return np.random.normal(0, 0.3, constants['N'])

    def initial_O(constants):
        return np.random.normal(0, 0.2, constants['N'])

    initial_state = {
        'I': initial_I,
        'O': initial_O,
        'A': 1
    }

    def update_I_A(nodes, constants):
        node = nodes[0]
        nb = np.random.choice(model.get_neighbors(node))
        if abs(model.get_node_state(node, 'O') - model.get_node_state(nb, 'O')) > constants['t_O']:
            return {'I': model.get_node_state(node, 'I')}
        else:
            # Update information
            r = constants['r_min'] + (1 - constants['r_min']) / (1 + np.exp(-1 * constants['p'] * (model.get_node_state(node, 'O') - model.get_node_state(nb, 'O'))))
            inf = r * model.get_node_state(node, 'I') + (1-r) * model.get_node_state(nb, 'I') + np.random.normal(0, constants['s_I'])

            # Update attention
            node_A = model.get_node_state(node, 'A') + constants['d_A'] * (2 * constants['A_star'] - model.get_node_state(node, 'A'))
            nb_A = model.get_node_state(nb, 'A') + constants['d_A'] * (2 * constants['A_star'] - model.get_node_state(nb, 'A'))
            return {'I': [inf], 'A': {node: node_A, nb: nb_A}}

    def update_A(constants):
        return {'A': model.get_state('A') - 2 * constants['d_A'] * model.get_state('A')/constants['N']}

    def update_O(constants):
        noise = np.random.normal(0, constants['s_O'], constants['N'])
        x = model.get_state('O') - constants['dt'] * (model.get_state('O')**3 - (model.get_state('A') + constants['A_min']) * model.get_state('O') - model.get_state('I')) + noise
        return {'O': x}

    def shrink_I():
        return {'I': model.get_state('I') * 0.999}

    def shrink_A():
        return {'A': model.get_state('A') * 0.999}

    def sample_attention_weighted(graph):
        probs = []
        A = model.get_state('A')
        factor = 1.0/sum(A)
        for a in A:
            probs.append(a * factor)
        return np.random.choice(graph.nodes, size=1, replace=False, p=probs)

    # Model definition
    model.constants = constants
    model.set_states(['I', 'A', 'O'])

    update_cfg = UpdateConfiguration({
        'arguments': {'constants': model.constants},
        'get_nodes': True
    })
    up_I_A = Update(update_I_A, update_cfg)
    s_I = Update(shrink_I)
    s_A = Update(shrink_A)

    model.add_scheme(Scheme(sample_attention_weighted, {'args': {'graph': model.graph}, 'updates': [up_I_A]}))
    model.add_scheme(Scheme(lambda graph: graph.nodes, {'args': {'graph': model.graph}, 'lower_bound': 5000, 'updates': [s_I]}))
    model.add_scheme(Scheme(lambda graph: graph.nodes, {'args': {'graph': model.graph}, 'lower_bound': 10000, 'updates': [s_A]}))
    model.add_update(update_A, {'constants': model.constants})
    model.add_update(update_O, {'constants': model.constants})

    model.set_initial_state(initial_state, {'constants': model.constants})

    iterations = model.simulate(15000)

    visualization_config = {
        'layout': 'fr',
        'plot_interval': 100,
        'plot_variable': 'O',
        'variable_limits': {
            'A': [0, 1],
            'O': [-1, 1],
            'I': [-1, 1]
        },
        'cmin': -1,
        'cmax': 1,
        'color_scale': 'RdBu',
        'show_plot': True,
        # 'plot_output': '../animations/HIOM.gif',
        'plot_title': 'HIERARCHICAL ISING OPINION MODEL',
    }

    model.configure_visualization(visualization_config, iterations)
    model.visualize('animation')