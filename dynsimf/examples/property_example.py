import networkx as nx

from dynsimf.models.Model import Model
from dynsimf.models.components.PropertyFunction import PropertyFunction

if __name__ == '__main__':

    g = nx.random_geometric_graph(50, 0.3)
    model = Model(g)

    def node_amount(G):
        return len(G.nodes())

    prop1 = PropertyFunction('1', nx.average_clustering, 2, {'G': model.graph})
    prop2 = PropertyFunction('2', node_amount, 2, {'G': model.graph})

    model.add_property_function(prop1)
    model.add_property_function(prop2)

    model.simulate(7)
    out = model.get_properties()
    print(out)