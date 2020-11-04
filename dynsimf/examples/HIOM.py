import networkx as nx
import numpy as np

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration
from dynsimf.models.Memory import MemoryConfiguration
from dynsimf.models.Memory import MemoryConfigurationType
from dynsimf.models.Scheme import Scheme
from dynsimf.models.Update import Update
from dynsimf.models.Update import UpdateConfiguration
from dynsimf.models.Example import Example


class HIOM(Example):

    def __init__(self):
        n = 400

        g = nx.watts_strogatz_graph(n, 2, 0.02)

        cfg = {
            'utility': False,
        }
        self.model = Model(g, ModelConfiguration(cfg))

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
            nb = np.random.choice(self.model.get_neighbors(node))
            if abs(self.model.get_node_state(node, 'O') - self.model.get_node_state(nb, 'O')) > constants['t_O']:
                return {'I': self.model.get_node_state(node, 'I')}
            else:
                # Update information
                r = constants['r_min'] + (1 - constants['r_min']) / (1 + np.exp(-1 * constants['p'] * (self.model.get_node_state(node, 'O') - self.model.get_node_state(nb, 'O'))))
                inf = r * self.model.get_node_state(node, 'I') + (1-r) * self.model.get_node_state(nb, 'I') + np.random.normal(0, constants['s_I'])

                # Update attention
                node_A = self.model.get_node_state(node, 'A') + constants['d_A'] * (2 * constants['A_star'] - self.model.get_node_state(node, 'A'))
                nb_A = self.model.get_node_state(nb, 'A') + constants['d_A'] * (2 * constants['A_star'] - self.model.get_node_state(nb, 'A'))
                return {'I': [inf], 'A': {node: node_A, nb: nb_A}}

        def update_A(constants):
            return {'A': self.model.get_state('A') - 2 * constants['d_A'] * self.model.get_state('A')/constants['N']}

        def update_O(constants):
            noise = np.random.normal(0, constants['s_O'], constants['N'])
            x = self.model.get_state('O') - constants['dt'] * (self.model.get_state('O')**3 - (self.model.get_state('A') + constants['A_min']) * self.model.get_state('O') - self.model.get_state('I')) + noise
            return {'O': x}

        def shrink_I():
            return {'I': self.model.get_state('I') * 0.999}

        def shrink_A():
            return {'A': self.model.get_state('A') * 0.999}

        def sample_attention_weighted(graph):
            probs = []
            A = self.model.get_state('A')
            factor = 1.0/sum(A)
            for a in A:
                probs.append(a * factor)
            return np.random.choice(graph.nodes, size=1, replace=False, p=probs)

        # Model definition
        self.model.constants = constants
        self.model.set_states(['I', 'A', 'O'])

        update_cfg = UpdateConfiguration({
            'arguments': {'constants': self.model.constants},
            'get_nodes': True
        })
        up_I_A = Update(update_I_A, update_cfg)
        s_I = Update(shrink_I)
        s_A = Update(shrink_A)

        self.model.add_scheme(Scheme(sample_attention_weighted, {'graph': self.model.graph}, updates=[up_I_A]))
        self.model.add_scheme(Scheme(lambda graph: graph.nodes, {'graph': self.model.graph}, lower_bound=5000, updates=[s_I]))
        self.model.add_scheme(Scheme(lambda graph: graph.nodes, {'graph': self.model.graph}, lower_bound=10000, updates=[s_A]))
        self.model.add_update(update_A, {'constants': self.model.constants})
        self.model.add_update(update_O, {'constants': self.model.constants})

        self.model.set_initial_state(initial_state, {'constants': self.model.constants})

    def simulate(self, n):
        return self.model.simulate(n)

    def visualize(self, iterations):
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

        self.model.configure_visualization(visualization_config, iterations)
        self.model.visualize('animation')
