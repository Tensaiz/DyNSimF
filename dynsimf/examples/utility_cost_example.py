import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from dynsimf.models.UtilityCostModel import FunctionType
from dynsimf.models.UtilityCostModel import SampleMethod
from dynsimf.models.UtilityCostModel import UtilityCostModel
from dynsimf.models.Model import ModelConfiguration
from dynsimf.models.components.Memory import MemoryConfiguration
from dynsimf.models.components.Memory import MemoryConfigurationType


if __name__ == "__main__":
    # Network definition
    n_nodes = 50
    g = nx.random_geometric_graph(n_nodes, 0.2)

    cfg = {
        'adjacency_memory_config': \
            MemoryConfiguration(MemoryConfigurationType.ADJACENCY, {
                'memory_size': 0
            }),
        'utility_memory_config': \
            MemoryConfiguration(MemoryConfigurationType.UTILITY, {
                'memory_size': 0
            })
    }
    cost_threshold = 1
    model = UtilityCostModel(g, cost_threshold, ModelConfiguration(cfg))

    constants = {
        'q': 0.8,
        'b': 0.5,
        'd': 0.2,
        'h': 0.2,
        'k': 0.25,
        'S+': 0.5,
        'P': np.random.random_sample(n_nodes)
    }
    constants['p'] = 2*constants['d']

    def initial_v(constants):
        return np.minimum(1, np.maximum(0, model.get_state('C') - model.get_state('S') - model.get_state('E')))

    def initial_a(constants):
        return constants['q'] * model.get_state('V') + (np.random.poisson(model.get_state('lambda')) / 4)

    initial_state = {
        'C': 0,
        'S': constants['S+'],
        'E': 0.5,
        'V': initial_v,
        'lambda': 0.5,
        'A': initial_a,
    }

    def update_C(constants):
        c = model.get_state('C') + constants['b'] * model.get_state('A') * np.minimum(1, 1-model.get_state('C')) - constants['d'] * model.get_state('C')
        return {'C': c}

    def update_S(constants):
        return {'S': model.get_state('S') + constants['p'] * np.maximum(0, constants['S+'] - model.get_state('S')) - constants['h'] * model.get_state('C') - constants['k'] * model.get_state('A')}

    def update_E(constants):
        adj = model.get_adjacency()
        summed = np.matmul(adj, model.get_nodes_states())
        sums_addiction = summed[:, model.get_state_index('A')] / 10

        change =  model.get_state('E') + (0.2 - sums_addiction) / 10

        return {'E': np.minimum(np.maximum(-1.5, change), 1)}

    def update_V(constants):
        return {'V': np.minimum(1, np.maximum(0, model.get_state('C')-model.get_state('S')-model.get_state('E')))}

    def update_A(constants):
        return {'A': constants['q'] * model.get_state('V') + np.minimum((np.random.poisson(model.get_state('lambda'))/2), constants['q']*(1 - model.get_state('V')))}

    # Model definition
    model.constants = constants
    model.set_states(['C', 'S', 'E', 'V', 'lambda', 'A'])
    model.add_update(update_C, {'constants': model.constants})
    model.add_update(update_S, {'constants': model.constants})
    model.add_update(update_E, {'constants': model.constants})
    model.add_update(update_V, {'constants': model.constants})
    model.add_update(update_A, {'constants': model.constants})
    model.set_initial_state(initial_state, {'constants': model.constants})

    def utility_calculation():
        utility = np.zeros((n_nodes, n_nodes))
        addiction = model.get_state('A')
        utility = (1 - abs(addiction[..., None] - addiction))
        np.fill_diagonal(utility, 0)

        return utility

    def cost_calculation():
        adjacency = model.get_adjacency()
        addiction = model.get_state('A')

        # Base cost
        cost = np.ones((n_nodes, n_nodes)) * 0.1
        cost = addiction[..., None] + cost
        extra_cost = (1 - adjacency) * 0.1
        cost = cost + extra_cost

        return cost

    model.add_utility_function(utility_calculation, FunctionType.MATRIX)
    model.add_cost_function(cost_calculation, FunctionType.MATRIX)
    model.set_sampling_function(SampleMethod.NEIGHBORS_OF_NEIGHBORS)
    # model.set_sampling_function(SampleMethod.ALL)

    output = model.simulate(100)

    state_values = output['states'].values()

    A = [np.mean(sv[:, 5]) for sv in state_values]
    C = [np.mean(sv[:, 0]) for sv in state_values]

    E = [np.mean(sv[:, 2]) for sv in state_values]
    lmd = [np.mean(sv[:, 4]) for sv in state_values]

    S = [np.mean(sv[:, 1]) for sv in state_values]
    V = [np.mean(sv[:, 3]) for sv in state_values]

    x = np.arange(0, len(state_values))
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

    visualization_config = {
        'plot_interval': 2,
        'initial_positions': nx.get_node_attributes(g, 'pos'),
        'plot_variable': 'A',
        # 'plot_variable': 'utility',
        'color_scale': 'Reds',
        'variable_limits': {
            'A': [0, 0.8],
            'lambda': [0.5, 1.5],
            'C': [-1, 1],
            'V': [-1, 1],
            'E': [-1, 1],
            'S': [-1, 1],
            'utility': [0, 25],
        },
        # 'show_plot': True,
        'repeat': True,
        'save_fps': 2,
        # 'plot_output': '../animations/c_vs_s_addiction.gif',
        'plot_title': 'Self control vs craving simulation'
    }

    model.configure_visualization(visualization_config, output)
    model.visualize('animation')
