import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration
from dynsimf.models.Example import Example


class CravingSelfControl(Example):

    def __init__(self):
        # Network definition
        g = nx.random_geometric_graph(250, 0.125)
        cfg = {
            'utility': False,
        }
        self.model = Model(g, ModelConfiguration(cfg))

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
            return np.minimum(1, np.maximum(0, self.model.get_state('C') - self.model.get_state('S') - self.model.get_state('E')))

        def initial_a(constants):
            return constants['q'] * self.model.get_state('V') + (np.random.poisson(self.model.get_state('lambda'))/7)

        initial_state = {
            'C': 0,
            'S': constants['S+'],
            'E': 1,
            'V': initial_v,
            'lambda': 0.5,
            'A': initial_a
        }

        def update_C(constants):
            c = self.model.get_state('C') + constants['b'] * self.model.get_state('A') * np.minimum(1, 1-self.model.get_state('C')) - constants['d'] * self.model.get_state('C')
            return {'C': c}

        def update_S(constants):
            return {'S': self.model.get_state('S') + constants['p'] * np.maximum(0, constants['S+'] - self.model.get_state('S')) - constants['h'] * self.model.get_state('C') - constants['k'] * self.model.get_state('A')}

        # Naive manner
        # def update_E(constants):
        #     # return {'E': self.model.get_state('E') - 0.015}
        #     e = np.zeros(len(self.model.nodes))
        #     for i, node in enumerate(self.model.nodes):
        #         neighbor_addiction = 0
        #         for neighbor in self.model.get_neighbors(node):
        #             neighbor_addiction += self.model.get_node_state(neighbor, 'A')
        #         e[i] = neighbor_addiction / 50
        #     return {'E': np.maximum(-1.5, self.model.get_state('E') - e)} # Custom calculation

        # Less naive
        # def update_E(constants):
        #     e = np.zeros(len(self.model.nodes))
        #     adj = self.model.get_adjacency()
        #     for i in range(len(self.model.nodes)):
        #         neighbors = adj[i].nonzero()
        #         e[i] = np.sum(self.model.get_nodes_state(neighbors, 'A')) / 50
        #     return {'E': np.maximum(-1.5, self.model.get_state('E') - e)} # Custom calculation

        def update_E(constants):
            adj = self.model.get_adjacency()
            summed = np.matmul(adj, self.model.get_nodes_states())
            e = summed[:, self.model.get_state_index('A')] / 50
            return {'E': np.maximum(-1.5, self.model.get_state('E') - e)}

        def update_V(constants):
            return {'V': np.minimum(1, np.maximum(0, self.model.get_state('C')-self.model.get_state('S')-self.model.get_state('E')))}

        def update_lambda(constants):
            return {'lambda': self.model.get_state('lambda') + 0.01}

        def update_A(constants):
            return {'A': constants['q'] * self.model.get_state('V') + np.minimum((np.random.poisson(self.model.get_state('lambda'))/7), constants['q']*(1 - self.model.get_state('V')))}

        # Model definition
        self.model.constants = constants
        self.model.set_states(['C', 'S', 'E', 'V', 'lambda', 'A'])
        self.model.add_update(update_C, {'constants': self.model.constants})
        self.model.add_update(update_S, {'constants': self.model.constants})
        self.model.add_update(update_E, {'constants': self.model.constants})
        self.model.add_update(update_V, {'constants': self.model.constants})
        self.model.add_update(update_lambda, {'constants': self.model.constants})
        self.model.add_update(update_A, {'constants': self.model.constants})
        self.model.set_initial_state(initial_state, {'constants': self.model.constants})

    def simulate(self, n):
        return self.model.simulate(n)

    def plot_paper(self, iterations):
        A = [np.mean(it[:, 5]) for it in iterations]
        C = [np.mean(it[:, 0]) for it in iterations]

        E = [np.mean(it[:, 2]) for it in iterations]
        lmd = [np.mean(it[:, 4]) for it in iterations]

        S = [np.mean(it[:, 1]) for it in iterations]
        V = [np.mean(it[:, 3]) for it in iterations]

        x = np.arange(0, len(iterations))
        plt.figure()

        plt.subplot(221)
        plt.plot(x, E, label='E')
        plt.plot(x, lmd, label='lambda')
        plt.legend()

        plt.subplot(222)
        plt.plot(x, A, label='A')
        plt.plot(x, C, label='C')
        plt.legend()

        plt.subplot(223)
        plt.plot(x, S, label='S')
        plt.plot(x, V, label='V')
        plt.legend()

        plt.show()

    def visualize(self, iterations):
        visualization_config = {
            'plot_interval': 2,
            'plot_variable': 'A',
            'color_scale': 'RdBu',
            'variable_limits': {
                'A': [0, 0.8],
                'lambda': [0.5, 1.5],
                'C': [-1, 1],
                'V': [-1, 1],
                'E': [-1, 1],
                'S': [-1, 1]
            },
            'show_plot': True,
            # 'plot_output': './animations/c_vs_s.gif',
            'plot_title': 'Self control vs craving simulation',
        }

        self.model.configure_visualization(visualization_config, iterations)
        self.model.visualize('animation')
