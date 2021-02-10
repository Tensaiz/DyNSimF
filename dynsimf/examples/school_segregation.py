import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration
from dynsimf.models.components.Memory import MemoryConfiguration
from dynsimf.models.components.Memory import MemoryConfigurationType
from dynsimf.models.components.conditions.Condition import ConditionType
from dynsimf.models.components.conditions.ThresholdCondition import ThresholdCondition
from dynsimf.models.components.conditions.CustomCondition import CustomCondition
from dynsimf.models.components.conditions.ThresholdCondition import ThresholdOperator
from dynsimf.models.components.conditions.ThresholdCondition import ThresholdConfiguration

if __name__ == "__main__":
    # Network definition
    g_list = pickle.load(open(r"C:/Users/Admin/MEGA/Uni/Master/Thesis/data/g_list.pkl", 'rb'))
    X_list = pickle.load(open(r"C:/Users/Admin/MEGA/Uni/Master/Thesis/data/x_list.pkl", 'rb'))

    school = 3

    X = X_list[school]

    n = len(X['sex'])
    avg_initial_links = 5 # desired average degree in initial network
    link_prop = avg_initial_links/n

    g = np.random.choice([0, 1], size=(n, n),
                               p=[1 - link_prop,
                               link_prop])
    np.fill_diagonal(g, 0)
    g = nx.convert_matrix.from_numpy_array(g, create_using=nx.DiGraph)

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
        'n': n,

        'delta': 0.05,
        'gamma': 0.65,
        'c': 0.175,
        'B1': 0.1,
        'B2': 0.1,
        'B3': 0.2,
        'sigma': 0.035,
        'alpha': 2,

        'min_prop': 1000,

        'X': X
    }

    def initial_utility():
        utility = np.zeros((constants['n'], constants['n']))

        race = list(constants['X']['race'])
        sex = list(constants['X']['sex'])
        grade = list(constants['X']['grade'])

        for i in range(constants['n']):
            for j in range(constants['n']):
                weighted_diffs = [constants['B1']*abs(sex[i] - sex[j]),
                                          constants['B2'] * (0 if grade[i] == grade[j] else 1),
                                          constants['B3'] * (0 if race[i] == race[j] else 1)]
                utility[i, j] = math.exp(-sum(weighted_diffs))

        return utility

    def initial_prop():
        prop = np.zeros((constants['n'], constants['n']))
        utility = initial_utility()

        # Loop over the person and their peers
        for i in range(constants['n']):
            for j in range(constants['n']):
                if i == j:
                    prop[i, j] = 0
                else:
                    prop[i, j] = utility[i, j] + constants['min_prop']

            # Normalize
            prop[i, :] = prop[i, :] / np.sum(prop[i, :])

        return prop

    constants['probability'] = initial_prop()
    constants['utility'] = initial_utility()

    def nb_update():
        adj = model.get_adjacency()

        return {'Neighbors': np.sum(adj, axis=1)}

    def node_utility(node, adj):
        utility = constants['utility']

        # degree, connection gain and cost calculations
        d_i = adj[node].sum()
        direct_u = np.sum(adj[node] * utility[node])
        mutual_u = np.sum(adj[node] * adj.T[node] * utility[node])

        # indirect connection gain
        a = (adj.T.dot(adj[node, :]) * utility)[node]
        a[node] = 0
        indirect_u = np.sum(a)

        return direct_u + constants['gamma'] * mutual_u + constants['delta'] * indirect_u - d_i ** constants['alpha'] * constants['c']

    def network_update(nodes):
        adj = model.get_adjacency()
        order = nodes.copy()

        eps = np.random.normal(scale=constants['sigma'], size=constants['n']*2)

        np.random.shuffle(order)

        changes = {}

        P = constants['probability']

        for node in order:

            other_node = node
            while other_node == node:
                other_node = np.random.choice(nodes, p=P[node])

            existing_connection = not not adj[node, other_node]

            adj[node, other_node] = 0
            U_without = node_utility(node, adj) + eps[node]

            adj[node, other_node] = 1
            U_with = node_utility(node, adj) + eps[-node]

            if U_without > U_with and existing_connection:
                changes[node] = {'remove': [other_node]}
            elif U_without < U_with and not existing_connection:
                changes[node] = {'add': [other_node]}

        return {
            'edge_change': changes
        }


    # Model definition
    model.constants = constants
    model.set_states(['Neighbors'])
    model.add_update(nb_update)
    model.set_edge_values(['utility'])
    model.set_initial_edge_values({
        'utility': initial_utility,
    })

    model.add_network_update(network_update, get_nodes=True)

    output = model.simulate(500)

    visualization_config = {
        'plot_interval': 10,
        'edge_values': 'utility',
        'plot_variable': 'Neighbors',
        'variable_limits': {
            'Neighbors': [0, 55]
        },
        'color_scale': 'Reds',
        'show_plot': False,
        'repeat': True,
        'plot_output': '../animations/school_segregation/school_' + str(school) + '.gif',
        'plot_title': 'School segregation'
    }

    model.configure_visualization(visualization_config, output)
    model.visualize('animation')
