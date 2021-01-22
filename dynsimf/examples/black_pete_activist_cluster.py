import networkx as nx
import numpy as np

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration
from dynsimf.models.components.Scheme import Scheme
from dynsimf.models.components.Update import Update
from dynsimf.models.components.Update import UpdateType
from dynsimf.models.components.Update import UpdateConfiguration
from dynsimf.models.components.Memory import MemoryConfiguration
from dynsimf.models.components.Memory import MemoryConfigurationType


if __name__ == "__main__":
    n = 50
    clusters = 1
    p_within = 1
    p_between = 1
    rewiring = .02

    sizes = list(map(int, [10/clusters] * clusters))
    pm = np.ones((clusters, clusters)) * p_between
    np.fill_diagonal(pm, p_within)
    g = nx.stochastic_block_model(sizes, pm)

    for i in range(10, 50):
        g.add_node(i)

    for i in range(10, 50):
        g.add_edge(i, np.random.randint(0, 50))

    cfg = {
        'utility': False,
        'adjacency_memory_config': \
            MemoryConfiguration(MemoryConfigurationType.ADJACENCY, {
                'memory_size': 0
        }),
    }
    model = Model(g, ModelConfiguration(cfg))

    constants = {
        'max_links': 20,

        'N': n,
        'sd_noise_information': .005,
        'persuasion': 2,
        'r_min': 0.1,
        's_O': .01,
        'maxwell_convention': False,
        'attention_star': 1,
        'min_attention': -.5, # to include continuous change in O as function of K
        'delta_attention': 0.1,
        'decay_attention': 0.1/(1*(n^2)),
        'deffuant_c': np.inf
    }

    def stoch_cusp(N,x,b,a,s_O,maxwell_convention=False):
        dt=.01
        x = x-dt*(x**3-b*x-a)+np.random.normal(0,s_O,N)
        np.nan_to_num(x)
        return x

    init_I = np.random.normal(.1, 0, constants['N'])
    init_O = np.random.normal(0, 0.01, constants['N'])
    for _ in range(500):
        init_O = stoch_cusp(constants['N'], init_O, np.zeros(constants['N'])+constants['min_attention'], init_I, constants['s_O'], False)

    # Activism
    m = np.zeros(constants['N'])
    m[0:10] = 1
    information_activists=-1
    opinion_activists=-1

    init_I[m == 1] = information_activists
    init_O[m == 1] = opinion_activists

    initial_state = {
        'I': init_I,
        'O': init_O,
    }

    def sample_attention_weighted(graph):
        A = get_normalized_links()
        if max(A) == 0:
            return []
        else:
            factor = 1.0/np.sum(A)
            probs = A * factor
            return np.random.choice(graph.nodes, size=1, replace=False, p=probs)

    def get_normalized_links():
        adj = model.get_adjacency()
        link_n = np.sum(adj, axis=1)
        links_norm = link_n / (constants['max_links']/1.5)
        return links_norm

    def update(constants):
        information = model.get_state('I')
        attention = get_normalized_links()
        opinion = model.get_state('O')

        if max(attention) == 0:
            agent = -1
        else:
            factor = 1.0/np.sum(attention)
            probs = attention * factor
            agent = np.random.choice(list(range(constants['N'])), 1, replace=False, p=probs)[0]

        if agent != -1:
            neighbors = model.get_neighbors(agent)
            if len(neighbors) > 0:
                partner = np.random.choice(neighbors)
            else:
                partner = -1
        else:
            partner = -1

        if partner != -1 and agent != -1:
            agent_neighbors = model.get_neighbors(agent).append(partner)
            agent_neighbors_information = information[agent_neighbors]
            partner_neighbors = model.get_neighbors(partner).append(agent)
            partner_neighbors_information = information[partner_neighbors]

            information[agent] = np.average(np.append(agent_neighbors_information, information[agent]))
            information[partner] = np.average(np.append(partner_neighbors_information, information[partner]))

        information = information+np.random.normal(0,constants['sd_noise_information'],constants['N'])
        opinion = stoch_cusp(constants['N'],opinion,attention+constants['min_attention'],information,constants['s_O'], constants['maxwell_convention'])

        return {'O': opinion, 'I': information, 'A': attention}

    def network_update(nodes, constants):
        if len(nodes) > 0:
            node = nodes[0]
        else:
            node = np.random.randint(0, constants['N'])

        changes = {}
        update = {}

        adj = model.get_adjacency()

        # Agent creates a new connection
        if np.sum(adj[node]) < constants['max_links'] - 1 and np.random.rand() <= 0.5:
            potential_links = np.where(adj[node] == 0)[0]
            potential_links = potential_links[np.where(np.sum(adj[potential_links], axis=1) < constants['max_links'])[0]]
            if len(potential_links) > 0:
                new_link = np.random.choice(potential_links, 2)
                update['add'] = new_link
                changes[node] = update

        # remove x percent of links
        if model.current_iteration <= 2500 and np.random.rand() <= 0.25:
            A = 1.5 - get_normalized_links()
            if max(A) == 0:
                return []
            else:
                factor = 1.0/np.sum(A)
                probs = (A * factor)
                node = np.random.choice(np.arange(0, constants['N']), size=1, replace=False, p=probs)[0]

                # Remove a link from an agent
                neighbors = np.where(adj[node] == 1)[0]
                if len(neighbors) >= 1:
                    removable_link = np.random.choice(neighbors, 1)
                    changes[node] = {'remove': removable_link}

        return {
            'edge_change': changes
        }

    # Model definition
    model.constants = constants
    model.set_states(['I', 'A', 'O'])

    model.add_update(update, {'constants': model.constants})

    update_network_cfg = UpdateConfiguration({
        'arguments': {'constants': model.constants},
        'get_nodes': True,
        'update_type': UpdateType.NETWORK
    })
    update_network = Update(network_update, update_network_cfg)
    model.add_scheme(Scheme(sample_attention_weighted, {'args': {'graph': model.graph}, 'updates': [update_network]}))

    model.set_initial_state(initial_state, {'constants': model.constants})

    output = model.simulate(2500)

    init_graph = nx.convert_matrix.from_numpy_array(output['adjacency'][0], create_using=nx.DiGraph)
    fixed_positions = nx.drawing.spring_layout(init_graph)

    visualization_config = {
        # 'layout': nx.drawing.layout.spring_layout,
        'fixed_positions': fixed_positions,
        'plot_interval': 100,
        'plot_variable': 'O',
        'variable_limits': {
            'A': [0, 1.5],
            'O': [-1, 1],
            'I': [-1, 1]
        },
        'cmin': -1,
        'cmax': 1,
        'color_scale': 'RdBu',
        'show_plot': False,
        'directed': False,
        'plot_output': '../animations/Black_pete_cluster_2.gif',
        'plot_title': 'HIERARCHICAL ISING OPINION MODEL',
    }

    model.configure_visualization(visualization_config, output)
    model.visualize('animation')
