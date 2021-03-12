
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration

if __name__ == "__main__":
    # Network definition
    n = 200
    # g = nx.erdos_renyi_graph(n, 0.1)
    g = nx.random_geometric_graph(n, 0.3)

    model = Model(g)

    constants = {
        'n': n,
        'beta': 0.001,
        'gamma': 0.01,
        'percentage_infected': 0.01
    }


    def initial_infected(constants):
        number_of_initial_infected = constants['n'] * constants['percentage_infected']
        state = np.zeros(constants['n'])

        sampled_nodes = np.random.choice(np.arange(constants['n']), int(number_of_initial_infected), replace=False)
        state[sampled_nodes] = 1
        return state

    initial_state = {
        'state': initial_infected
    }

    # Matrix/array version
    def update_state(constants):
        state = model.get_state('state')
        adjacency = model.get_adjacency()

        infected_indices = np.where(state == 1)[0]

        infected_adj = adjacency[infected_indices]

        susceptibles = np.zeros(len(state))
        susceptibles[state==0] = 1

        susceptible_adjacency = susceptibles * infected_adj
        susceptible_neighbors = np.where(susceptible_adjacency==1)[1]

        infected_sampled = (np.random.random_sample(len(susceptible_neighbors)) < constants['beta'])
        new_infected_indices = np.unique(infected_sampled * susceptible_neighbors)

        if (0 not in susceptible_neighbors or not infected_sampled[0]) and len(new_infected_indices) > 0:
            new_infected_indices = np.delete(new_infected_indices, [0])

        state[new_infected_indices] = 1

        # Update infected to recovered
        recovery_chances = np.random.random_sample(len(infected_indices))
        new_states = (recovery_chances < constants['gamma']) * 2
        new_states[new_states == 0] = 1
        state[infected_indices] = new_states

        return {'state': state}

    # # For loop version
    # def update_state(constants):
    #     state = model.get_state('state')

    #     infected_indices = np.where(state == 1)[0]

    #     # Update infected neighbors to infect neighbors
    #     for infected in infected_indices:
    #         nbs = model.get_neighbors(infected)
    #         for nb in nbs:
    #             if state[nb] == 0 and np.random.random_sample() < constants['beta']:
    #                 state[nb] = 1

    #     # Update infected to recovered
    #     recovery_chances = np.random.random_sample(len(infected_indices))
    #     new_states = (recovery_chances < constants['gamma']) * 2
    #     new_states[new_states == 0] = 1
    #     state[infected_indices] = new_states
    
    #     return {'state': state}

    # Model definition
    model.constants = constants
    model.set_states(['state'])
    model.add_update(update_state, {'constants': model.constants})

    model.set_initial_state(initial_state, {'constants': model.constants})

    its = model.simulate(600)

    iterations = its['states'].values()

    s = [np.count_nonzero(it == 0) for it in iterations]
    i = [np.count_nonzero(it == 1) for it in iterations]
    r = [np.count_nonzero(it == 2) for it in iterations]

    x = np.arange(0, len(iterations))
    plt.figure()

    plt.plot(x, s, label='S')
    plt.plot(x, i, label='I')
    plt.plot(x, r, label='R')
    plt.legend()

    plt.show()

    visualization_config = {
        'initial_positions': nx.get_node_attributes(g, 'pos'),
        'plot_interval': 5,
        'plot_variable': 'state',
        'color_scale': 'brg',
        'variable_limits': {
            'state': [0, 2],
        },
        'show_plot': True,
        'plot_output': '../animations/sir_geo.gif',
        'plot_title': 'SIR model',
    }

    model.configure_visualization(visualization_config, its)
    model.visualize('animation')