### This code imports the required packages in order to run the model. 
### It then defines the parameters, constants and derivatives necessary to run the model. 
### The model is initialised with states 'S', 'I' and 'R' and the initial state and constants are set. 
### A function is then defined to update the state of the model at each step of the simulation. 
### This section of the code then finishes with a simulation of the model and a resulting plot of the three states as a function of time.

import networkx as nx # used to create graphs
import numpy as np # used to work with arrays
import matplotlib.pyplot as plt # used to plot data
from dynsimf.models.Model import Model # used to simulate the model
from dynsimf.models.Model import ModelConfiguration # used to configure the model

if __name__ == "__main__":
    # Network definition
    n = 1 # number of nodes
    g = nx.random_geometric_graph(n, 1)  # create the graph
    model = Model(g) # instantiate a model with the created graph

    init_infected = 3 # initial infected nodes
    initial_state = { # set the initial state of the model
        'S': 1000 - init_infected, # Susceptible
        'I': init_infected, # Infected
        'R': 0, # Recovered
    }

    constants = {
        'N': 1000, # Total number of nodes
        'beta': 0.4, # Infection rate
        'gamma': 0.04, # Recovery rate
        'dt': 0.01, # Time step
    }

    def deriv(S, I, constants):  # Derive differential equations
        N = constants['N']
        beta = constants['beta']
        gamma = constants['gamma']

        dSdt = -beta * S * I / N # Susceptible differential
        dIdt = beta * S * I / N - gamma * I # Infected differential
        dRdt = gamma * I # Recovered differential
        return dSdt, dIdt, dRdt

    def update(constants): # function to update the state of the model 
        dt = constants['dt']
        S = model.get_state('S') # get number of suscpetible nodes
        I = model.get_state('I') # get number of infected nodes
        R = model.get_state('R') # get number of recovered nodes

        dSdt, dIdt, dRdt = deriv(S, I, constants) # calculate the derivatives

        return {
            # 'state': s + (0.01 * (-s + 1))
            'S': S + (dt * dSdt), # update susceptbile state
            'I': I + (dt * dIdt), # Update infected state
            'R': R + (dt * dRdt) # Update recovered state
        }

    # Model definition
    model.set_states(['S', 'I', 'R']) # set states
    model.add_update(update, {'constants': constants}) # add an update function with constants
    model.set_initial_state(initial_state) # set the initial state

    its = model.simulate(10000) # perform simulation for 10000 iterations
    iterations = list(its['states'].values()) # get the states after 10000 iterations

    s = [v[0][0] for v in iterations] # get the values of S
    i = [v[0][1] for v in iterations] # get the values of I 
    r = [v[0][2] for v in iterations] # get the values of R

    import matplotlib.pyplot as plt # import library 
    x = np.linspace(0, 100, 10001) # create x axis

    plt.figure() # create new figure
    plt.plot(x, s, label='S') # plot the S values against x axis
    plt.plot(x, i, label='I') # plot the I values against x axis
    plt.plot(x, r, label='R') # plot the R values against x axis
    plt.legend() # add legend
    plt.show() # show the plot
