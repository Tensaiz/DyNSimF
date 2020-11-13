import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration
from dynsimf.models.components.Memory import MemoryConfiguration
from dynsimf.models.components.Memory import MemoryConfigurationType
from dynsimf.models.components.conditions.Condition import ConditionType
from dynsimf.models.components.conditions.ThresholdCondition import ThresholdCondition
from dynsimf.models.components.conditions.CustomCondition import CustomCondition
from dynsimf.models.components.conditions.ThresholdCondition import ThresholdOperator
from dynsimf.models.components.conditions.ThresholdCondition import ThresholdConfiguration
from dynsimf.models.components.conditions.StochasticCondition import StochasticCondition


if __name__ == "__main__":
    # Network definition
    n_nodes = 50
    g = nx.random_geometric_graph(n_nodes, 0.2)

    cfg = {
        'adjacency_memory_config': \
            MemoryConfiguration(MemoryConfigurationType.ADJACENCY, {
                'memory_size': 0
            }),
        'edge_values_memory_config': \
            MemoryConfiguration(MemoryConfigurationType.EDGE_VALUES, {
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

    def calculate_utility():
        utility = np.zeros((n_nodes, n_nodes))
        adjacency = model.get_adjacency()
        addiction = model.get_state('A')

        for node, neighbors in enumerate(adjacency):
            node_addiction = addiction[node]
            neighbor_locations = np.where(neighbors == 1)[0]
            neighbors_addiction = addiction[neighbor_locations]
            neighbors_utility = 1 - abs(node_addiction - neighbors_addiction)

            utility[node, neighbor_locations] = neighbors_utility
        return {'addiction_utility': utility}

    def initial_utility():
        utility = np.zeros((n_nodes, n_nodes))
        adjacency = model.get_adjacency()
        addiction = model.get_state('A')

        for node, neighbors in enumerate(adjacency):
            node_addiction = addiction[node]
            neighbor_locations = np.where(neighbors == 1)[0]
            neighbors_addiction = addiction[neighbor_locations]
            neighbors_utility = 1 - abs(node_addiction - neighbors_addiction)

            utility[node, neighbor_locations] = neighbors_utility
        return utility

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

    def remove_neighbor(nodes):
        removable_neighbors = {}
        for node in nodes:
            removable_neighbors[node] = {'remove': []}
            neighbors = model.get_neighbors(node)

            f_utility = 1 - abs(constants['P'][neighbors] - constants['P'][node])
            # softmax
            weights = np.exp(f_utility) / np.sum(np.exp(f_utility), axis=0)
            removable_neighbors[node]['remove'].append(np.random.choice(neighbors, p=weights))
        return {
            'edge_change': removable_neighbors
        }

    def add_neighbor(nodes):
        A = model.get_state('A')
        addable_neighbors = {}
        for node in nodes:
            addable_neighbors[node] = {'add': []}
            neighbors_neighbors = model.get_neighbors_neighbors(node)
            if len(neighbors_neighbors) == 0:
                continue
            neighbors_neighbors_addiction = A[neighbors_neighbors]

            a_utility = 1 - abs(neighbors_neighbors_addiction - A[node])
            f_utility = 1 - abs(constants['P'][neighbors_neighbors] - constants['P'][node])
            weights = a_utility * f_utility

            # softmax
            weights = np.exp(weights) / np.sum(np.exp(weights), axis=0)
            selected_neighbor = np.random.choice(neighbors_neighbors, p=weights)
            selected_neighbor_index = np.where(neighbors_neighbors == selected_neighbor)[0]
            addable_neighbors[node]['add'].append((selected_neighbor, 'addiction_utility', a_utility[selected_neighbor_index], a_utility[selected_neighbor_index]))

        return {
            'edge_change': addable_neighbors
        }


    def add_condition(model_input):
        nodes = model_input[0]
        nodes_addiction = model.get_state('A')[nodes]
        # sigmoid (?)
        chances = (1 - nodes_addiction) / 10
        draws = np.random.random_sample(len(nodes))
        indices = np.where(draws < chances)[0]

        return nodes[indices]

    def remove_condition(model_input):
        nodes = model_input[0]
        nodes_addiction = model.get_state('A')[nodes]

        chances = nodes_addiction / 10
        draws = np.random.random_sample(len(nodes))
        indices = np.where(draws < chances)[0]

        return nodes[indices]

    # Model definition
    model.constants = constants
    model.set_states(['C', 'S', 'E', 'V', 'lambda', 'A'])
    model.add_update(update_C, {'constants': model.constants})
    model.add_update(update_S, {'constants': model.constants})
    model.add_update(update_E, {'constants': model.constants})
    model.add_update(update_V, {'constants': model.constants})
    model.add_update(update_A, {'constants': model.constants})

    model.set_edge_values(['addiction_utility'])
    model.set_initial_edge_values({
        'addiction_utility': initial_utility
    })
    model.add_edge_values_update(calculate_utility)

    condition_threshold_nb_cfg = ThresholdConfiguration(ThresholdOperator.GE, 1)

    add_c = CustomCondition(add_condition)
    condition_nb_a = ThresholdCondition(ConditionType.ADJACENCY, condition_threshold_nb_cfg, chained_condition=add_c)

    remove_c_stochastic = CustomCondition(remove_condition)
    # Only remove if there are neighbors
    condition_nb_r = ThresholdCondition(ConditionType.ADJACENCY, condition_threshold_nb_cfg, chained_condition=remove_c_stochastic)

    model.add_network_update(add_neighbor, condition=condition_nb_a, get_nodes=True)
    model.add_network_update(remove_neighbor, condition=condition_nb_r, get_nodes=True)

    model.set_initial_state(initial_state, {'constants': model.constants})

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
        'edge_values': 'addiction_utility',
        'initial_positions': nx.get_node_attributes(g, 'pos'),
        'plot_variable': 'A',
        'color_scale': 'Reds',
        'variable_limits': {
            'A': [0, 0.8],
            'lambda': [0.5, 1.5],
            'C': [-1, 1],
            'V': [-1, 1],
            'E': [-1, 1],
            'S': [-1, 1]
        },
        'show_plot': True,
        'repeat': True,
        'save_fps': 2,
        # 'plot_output': './animations/c_vs_s_dynamic.gif',
        'plot_title': 'Self control vs craving simulation'
    }

    model.configure_visualization(visualization_config, output)
    model.visualize('animation')
