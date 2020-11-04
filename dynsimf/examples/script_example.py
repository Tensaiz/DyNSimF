import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration
from dynsimf.models.Memory import MemoryConfiguration
from dynsimf.models.Memory import MemoryConfigurationType
from dynsimf.models.conditions.Condition import ConditionType
from dynsimf.models.conditions.ThresholdCondition import ThresholdCondition
from dynsimf.models.conditions.ThresholdCondition import ThresholdOperator
from dynsimf.models.conditions.ThresholdCondition import ThresholdConfiguration
from dynsimf.models.conditions.StochasticCondition import StochasticCondition


if __name__ == "__main__":
    # Network definition
    g = nx.random_geometric_graph(200, 0.125)

    cfg = {
        'utility': True,
        'adjacency_memory_config': \
            MemoryConfiguration(MemoryConfigurationType.ADJACENCY, {
                'memory_size': 0
            })
    }
    model = Model(g, ModelConfiguration(cfg))

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

    def inital_utility():
        n_nodes = len(g.nodes())
        return np.ones((n_nodes, n_nodes))

    def update_C(constants):
        c = model.get_state('C') + constants['b'] * model.get_state('A') * np.minimum(1, 1-model.get_state('C')) - constants['d'] * model.get_state('C')
        return {'C': c}

    def update_S(constants):
        return {'S': model.get_state('S') + constants['p'] * np.maximum(0, constants['S+'] - model.get_state('S')) - constants['h'] * model.get_state('C') - constants['k'] * model.get_state('A')}

    def update_E(constants):
        adj = model.get_adjacency()
        summed = np.matmul(adj, model.get_nodes_states())
        e = summed[:, model.get_state_index('A')] / 50
        return {'E': np.maximum(-1.5, model.get_state('E') - e)}

    def update_V(constants):
        return {'V': np.minimum(1, np.maximum(0, model.get_state('C')-model.get_state('S')-model.get_state('E')))}

    def update_lambda(constants):
        return {'lambda': model.get_state('lambda') + 0.01}

    def update_A(constants):
        return {'A': constants['q'] * model.get_state('V') + np.minimum((np.random.poisson(model.get_state('lambda'))/7), constants['q']*(1 - model.get_state('V')))}

    def update_utility():
        return model.get_utility()

    def remove_neighbor(nodes):
        adjacency = model.get_adjacency()

        removable_neighbors = {
            node: {'remove': [np.random.choice(np.where(adjacency[node] == 1)[0])]} for node in nodes
        }

        return {
            'edge_change': removable_neighbors
        }

    # Model definition
    model.constants = constants
    model.set_states(['C', 'S', 'E', 'V', 'lambda', 'A'])
    model.add_update(update_C, {'constants': model.constants})
    model.add_update(update_S, {'constants': model.constants})
    model.add_update(update_E, {'constants': model.constants})
    model.add_update(update_V, {'constants': model.constants})
    model.add_update(update_lambda, {'constants': model.constants})
    model.add_update(update_A, {'constants': model.constants})

    model.add_utility_update(update_utility)

    condition_stochastic_nb = StochasticCondition(ConditionType.STATE, 0.1)
    condition_threshold_nb_cfg = ThresholdConfiguration(ThresholdOperator.GE, 1)
    condition_nb = ThresholdCondition(ConditionType.ADJACENCY, condition_threshold_nb_cfg, chained_condition=condition_stochastic_nb)

    condition_threshold_cfg = ThresholdConfiguration(ThresholdOperator.GE, 0.75, state='A')
    condition_threshold_network = ThresholdCondition(ConditionType.STATE, condition_threshold_cfg, chained_condition=condition_nb)
    model.add_network_update(remove_neighbor, condition=condition_threshold_network, get_nodes=True)

    model.set_initial_state(initial_state, {'constants': model.constants})
    model.set_initial_utility(inital_utility)

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
        'plot_title': 'Self control vs craving simulation'
    }

    model.configure_visualization(visualization_config, output)
    model.visualize('animation')
