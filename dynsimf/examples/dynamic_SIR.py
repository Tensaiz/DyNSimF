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
    n = 250
    g = nx.random_geometric_graph(250, 0.125)

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


    def update_state(constants):
        state = model.get_state('state')

        infected_indices = np.where(state == 1)[0]

        # Update infected neighbors to infect neighbors
        for infected in infected_indices:
            nbs = model.get_neighbors(infected)
            for nb in nbs:
                if state[nb] == 0 and np.random.random_sample() < constants['beta']:
                    state[nb] = 1

        # Update infected to recovered
        recovery_chances = np.random.random_sample(len(infected_indices))
        new_states = (recovery_chances < constants['gamma']) * 2
        new_states[new_states == 0] = 1
        state[infected_indices] = new_states
    
        return {'state': state}

    # Model definition
    model.constants = constants
    model.set_states(['state'])
    model.add_update(update_state, {'constants': model.constants})

    model.set_initial_state(initial_state, {'constants': model.constants})

    def utility_calculation():
        states = model.get_state('state')
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
    # model.set_sampling_function(SampleMethod.NEIGHBORS_OF_NEIGHBORS)
    model.set_sampling_function(SampleMethod.ALL)

    its = model.simulate(400)

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
        'plot_interval': 5,
        'plot_variable': 'state',
        'color_scale': 'brg',
        'variable_limits': {
            'state': [0, 2],
        },
        # 'show_plot': True,
        'show_plot': True,
        'plot_output': '../animations/sir_dynamic.gif',
        'plot_title': 'Dynamic SIR model',
    }

    model.configure_visualization(visualization_config, output)
    model.visualize('animation')
