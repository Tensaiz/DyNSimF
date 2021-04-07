
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration

if __name__ == "__main__":
    # Network definition
    n = 1
    g = nx.random_geometric_graph(n, 1)
    model = Model(g)

    init_infected = 3
    initial_state = {
        'S': 1000 - init_infected,
        'I': init_infected,
        'R': 0,
    }

    constants = {
        'N': 1000,
        'beta': 0.4,
        'gamma': 0.04,
        'dt': 0.01,
    }

    def deriv(S, I, constants):
        N = constants['N']
        beta = constants['beta']
        gamma = constants['gamma']

        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def update(constants):
        dt = constants['dt']
        S = model.get_state('S')
        I = model.get_state('I')
        R = model.get_state('R')

        dSdt, dIdt, dRdt = deriv(S, I, constants)

        return {
            # 'state': s + (0.01 * (-s + 1))
            'S': S + (dt * dSdt),
            'I': I + (dt * dIdt),
            'R': R + (dt * dRdt)
        }

    # Model definition
    model.set_states(['S', 'I', 'R'])
    model.add_update(update, {'constants': constants})
    model.set_initial_state(initial_state)

    its = model.simulate(10000)
    iterations = list(its['states'].values())

    s = [v[0][0] for v in iterations]
    i = [v[0][1] for v in iterations]
    r = [v[0][2] for v in iterations]

    import matplotlib.pyplot as plt
    x = np.linspace(0, 100, 10001)

    plt.figure()
    plt.plot(x, s, label='S')
    plt.plot(x, i, label='I')
    plt.plot(x, r, label='R')
    plt.legend()
    plt.show()
