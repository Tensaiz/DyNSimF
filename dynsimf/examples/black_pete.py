import networkx as nx
import numpy as np

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration
from dynsimf.models.components.Scheme import Scheme
from dynsimf.models.components.Update import Update
from dynsimf.models.components.Update import UpdateConfiguration


if __name__ == "__main__":
    n = 400
    clusters = 10
    p_within = .2
    p_between = .001
    rewiring = .02

    sizes = list(map(int, [n/clusters] * clusters))
    pm = np.ones((10, 10)) * p_between
    np.fill_diagonal(pm, p_within)
    g = nx.stochastic_block_model(sizes, pm)

    cfg = {
        'utility': False,
    }
    model = Model(g, ModelConfiguration(cfg))

    constants = {
        'dt': 0.01,
        'A_min': -0.5,
        'A_star': 1,
        's_O': 0.01,
        's_I': 0.0005,
        'd_A': 0.01,
        'p': 2,
        'r_min': 0.1,
        'sd_noise_information': 0.005,
        't_O': np.inf,
        'N': n
    }

    def initial_I(constants):
        init_I = np.array([0.1] * constants['N'])
        return init_I

    def initial_O(constants):
        init_O = np.random.normal(0, 0.01, constants['N'])
        return init_O

    initial_state = {
        'I': initial_I,
        'O': initial_O,
        'A': 0.000001
    }

    def update_I_A(nodes, constants):
        node = nodes[0]
        nb = np.random.choice(model.get_neighbors(node))
        if abs(model.get_node_new_state(node, 'O') - model.get_node_new_state(nb, 'O')) > constants['t_O']:
            return {'I': model.get_node_new_state(node, 'I')}
        else:
            # Update information
            r = constants['r_min'] + (1 - constants['r_min']) / (1 + np.exp(-1 * constants['p'] * (model.get_node_new_state(node, 'O') - model.get_node_new_state(nb, 'O'))))
            inf = r * model.get_node_new_state(node, 'I') + (1-r) * model.get_node_new_state(nb, 'I') + np.random.normal(0, constants['s_I'])

            # Update attention
            node_A = model.get_node_new_state(node, 'A') + constants['d_A'] * (2 * constants['A_star'] - model.get_node_new_state(node, 'A'))
            nb_A = model.get_node_new_state(nb, 'A') + constants['d_A'] * (2 * constants['A_star'] - model.get_node_new_state(nb, 'A'))
            return {'I': [inf], 'A': {node: node_A, nb: nb_A}}

    def update_I(constants):
        return {'I': model.get_new_state('I') + np.random.normal(0, constants['sd_noise_information'], constants['N'])}

    def update_A(constants):
        return {'A': model.get_new_state('A') - 2 * constants['d_A'] * model.get_new_state('A')/constants['N']}

    def update_O(constants):
        noise = np.random.normal(0, constants['s_O'], constants['N'])
        x = model.get_new_state('O') - constants['dt'] * (model.get_new_state('O')**3 - (model.get_new_state('A') + constants['A_min']) * model.get_new_state('O') - model.get_new_state('I')) + noise
        return {'O': x}

    def activism_A():
        A = model.get_new_state('A')
        A[np.arange(1, constants['N'], constants['N']/3)[1:].astype(int)-1] = 1
        return {'A': A}

    def activism_I():
        I = model.get_new_state('I')
        I[np.arange(1, constants['N'], constants['N']/3)[1:].astype(int)-1] = -0.5
        return {'I': I}

    def activism_O():
        O = model.get_new_state('O')
        O[np.arange(1, constants['N'], constants['N']/3)[1:].astype(int)-1] = -0.5
        return {'O': O}

    def sample_attention_weighted(graph):
        probs = []
        A = model.get_new_state('A')
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

    model.add_update(update_I, {'constants': model.constants})
    model.add_update(update_A, {'constants': model.constants})
    model.add_update(update_O, {'constants': model.constants})
    model.add_scheme(Scheme(sample_attention_weighted, {'args': {'graph': model.graph}, 'updates': [up_I_A]}))

    a_I = Update(activism_I)
    a_A = Update(activism_A)
    a_O = Update(activism_O)

    model.add_scheme(Scheme(lambda graph: graph.nodes, {'args': {'graph': model.graph}, 'lower_bound': 300, 'upper_bound': 301, 'updates': [a_I, a_A, a_O]}))

    model.set_initial_state(initial_state, {'constants': model.constants})

    output = model.simulate(25000)

    visualization_config = {
        'layout': nx.drawing.layout.spring_layout,
        'plot_interval': 500,
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

    model.configure_visualization(visualization_config, output)
    model.visualize('animation')
