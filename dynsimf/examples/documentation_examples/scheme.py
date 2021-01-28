import networkx as nx
import numpy as np

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration
from dynsimf.models.components.Update import Update
from dynsimf.models.components.Update import UpdateType
from dynsimf.models.components.Update import UpdateConfiguration
from dynsimf.models.components.Scheme import Scheme

if __name__ == "__main__":

    g = nx.erdos_renyi_graph(n=10, p=0.1)
    model = Model(g, ModelConfiguration())

    # Define schemes
    def sample_state_weighted(graph):
        probs = []
        status_1 = model.get_state('status_1')
        factor = 1.0/sum(status_1)
        for s in status_1:
            probs.append(s * factor)
        return np.random.choice(graph.nodes, size=1, replace=False, p=probs)

    initial_state = {
        'status_1': 0.1,
        'status_2': 0.1,
    }

    model.set_states(['status_1', 'status_2'])

    # Update functions
    def update_1(nodes):
        node = nodes[0]
        node_update = model.get_state('status_2')[node] + 0.1
        return {'status_1': {node: node_update}}

    def update_2():
        s2 = model.get_state('status_2') + 0.1
        return {'status_2': s2}


    update_cfg = UpdateConfiguration({
        'arguments': {}, # No other arguments
        'get_nodes': True, # We want the nodes as argument
        'update_type': UpdateType.STATE
    })
    u = Update(update_1, update_cfg) # Create an Update object that contains the object function
    # Create a scheme with the correct sample function, parameters, bounds, and update function
    model.add_scheme(Scheme(sample_state_weighted, {'args': {'graph': model.graph}, 'lower_bound': 2, 'upper_bound': 3, 'updates': [u]}))

    # Add update 2 to the model, which will increase status_1 by 0.5 each iteration
    model.add_update(update_2)

    model.set_initial_state(initial_state)
    output = model.simulate(5)
    print(output)