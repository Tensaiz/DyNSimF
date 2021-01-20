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
    n = 400
    clusters = 10
    p_within = .2
    p_between = .001
    rewiring = .02

    sizes = list(map(int, [n/clusters] * clusters))
    pm = np.ones((10, 10)) * p_between
    np.fill_diagonal(pm, p_within)
    g = nx.stochastic_block_model(sizes, pm)
    fixed_positions = nx.drawing.spring_layout(g)
    g.remove_edges_from(list(g.edges))

    cfg = {
        'utility': False,
        'adjacency_memory_config': \
            MemoryConfiguration(MemoryConfigurationType.ADJACENCY, {
                'memory_size': 0
        }),
    }
    model = Model(g, ModelConfiguration(cfg))

    constants = {
        'link_addition_p': 0.75,
        'link_removal_p': 0.0005,
        'max_links': 15,

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

    def information_update(i1,i2,a1,a2,persuasion,r_min,info_update=True):
        r=r_min+(1-r_min)/(1+np.exp(-1*persuasion*(a1-a2))) # resistance
        inf=r*i1+(1-r)*i2  # compute weighted man of information
        if not info_update:
            inf = i1
        return inf

    init_I = np.random.normal(.1, 0, constants['N'])
    init_O = np.random.normal(0, 0.01, constants['N'])
    for _ in range(500):
        init_O = stoch_cusp(constants['N'], init_O, np.zeros(constants['N'])+constants['min_attention'], init_I, constants['s_O'], False)

    initial_state = {
        'I': init_I,
        'O': init_O,
        # 'A': 0
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
        links_norm = link_n / (15/1.5)
        # links_norm = links_n / constants['max_links']
        return links_norm

    def update(constants):
        information = model.get_state('I')
        # attention = model.get_state('A')
        attention = get_normalized_links()
        opinion = model.get_state('O')

        if max(attention) == 0:
            agent = -1
        else:
            factor = 1.0/np.sum(attention)
            probs = attention * factor
            agent = np.random.choice(list(range(constants['N'])), 1, replace=False, p=probs)

        if agent != -1:
            neighbors = model.get_neighbors(agent[0])
            if len(neighbors) > 0:
                partner = np.random.choice(neighbors)
            else:
                partner = -1
        else:
            partner = -1

        if partner != -1 and agent != -1:
            I1 = information[agent]; A1 = attention[agent]; O1 = opinion[agent]
            I2 = information[partner]; A2 = attention[partner]; O2 = opinion[partner]

            if abs(O1 - O2) < constants['deffuant_c']:
                information[agent] = information_update(I1,I2,A1,A2,constants['persuasion'],constants['r_min'],True)
                information[partner] = information_update(I2,I1,A2,A1,constants['persuasion'],constants['r_min'],True)

            # if abs(O1 - O2) < constants['deffuant_c']:
            #     attention[agent]=attention[agent]+constants['delta_attention']*(2*constants['attention_star']-attention[agent])
            #     attention[partner]=attention[partner]+constants['delta_attention']*(2*constants['attention_star']-attention[partner])

        information = information+np.random.normal(0,constants['sd_noise_information'],constants['N'])
        # attention = attention-2*constants['delta_attention']*attention/constants['N']   # correction 2 times if interaction
        opinion = stoch_cusp(constants['N'],opinion,attention+constants['min_attention'],information,constants['s_O'], constants['maxwell_convention'])

        return {'O': opinion, 'I': information, 'A': attention}

    def activism_states():
        information = model.get_state('I')
        opinion = model.get_state('O')

        m = np.zeros(constants['N'])
        m[np.arange(1, constants['N'], constants['N']/3)[1:].astype(int)-1] = 1

        information_activists=-1
        opinion_activists=-1

        information[m == 1] = information_activists
        opinion[m == 1] = opinion_activists

        return {'O': opinion, 'I': information}

    def activism_network():
        changes = {}
        activist_nodes = np.arange(1, constants['N'], constants['N']/3)[1:].astype(int)-1
        adj = model.get_adjacency()
        for node in activist_nodes:
            new_links = int(constants['max_links'] - sum(adj[node]))
            changes[node] = {'add': np.random.choice(np.where(adj[node] == 0)[0], new_links)}

        return {
            'edge_change': changes
        }

    def network_update(nodes, constants):
        if len(nodes) > 0:
            node = nodes[0]
        else:
            node = np.random.randint(0, constants['N'])

        changes = {}
        update = {}

        adj = model.get_adjacency()
        # nb_adj = model.get_neighbors_neighbors_adjacency_matrix()

        # chance for an agent to create a new connection
        if np.random.random_sample() <= constants['link_addition_p'] and np.sum(adj[node]) < constants['max_links']:
            # Pick a new link that does not exist yet for this agent
            # potential_links = np.where(nb_adj[node] == 1)[0] # Neighbors of neighbors
            potential_links = np.where(adj[node] == 0)[0]

            if len(potential_links) > 0:
                new_link = np.random.choice(potential_links)
                update['add'] = [new_link]
                changes[node] = update
            else:
                new_link = np.random.choice(np.where(adj[node] == 1)[0])
                update['add'] = [new_link]
                changes[node] = update

        # remove x percent of links
        total_links = np.sum(adj)
        removable_links_n = np.floor(total_links * constants['link_removal_p'])

        while removable_links_n > 0:
            # Pick a random node
            node = np.random.randint(0, constants['N'])
            # Pick a link that does exists yet for this agent
            neighbors = np.where(adj[node] == 1)[0]

            if len(neighbors) >= 1:
                remove_amount = np.random.randint(1, len(neighbors) + 1)
                removable_links = np.random.choice(neighbors, remove_amount)
                changes[node] = {'remove': removable_links}
                removable_links_n -= remove_amount

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

    activism_update = Update(activism_states)
    activism_network_update = Update(activism_network, UpdateConfiguration({'update_type': UpdateType.NETWORK}))
    model.add_scheme(Scheme(lambda graph: graph.nodes, {'args': {'graph': model.graph}, 'lower_bound': 300, 'upper_bound': 301, 'updates': [activism_update, activism_network_update]}))

    model.set_initial_state(initial_state, {'constants': model.constants})

    output = model.simulate(25000)

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
        'plot_output': '../animations/black_pete/unlinked_all_long.gif',
        'plot_title': 'HIERARCHICAL ISING OPINION MODEL',
    }

    model.configure_visualization(visualization_config, output)
    model.visualize('animation')
