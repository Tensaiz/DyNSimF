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
    np.random.seed(1)

    n = 400
    clusters = 10
    p_within = .2
    p_between = .001
    rewiring = .02

    sizes = list(map(int, [n/clusters] * clusters))
    pm = np.ones((10, 10)) * p_between
    np.fill_diagonal(pm, p_within)
    g = nx.stochastic_block_model(sizes, pm)

    cfg = {
        'utility': False,
        'adjacency_memory_config': \
            MemoryConfiguration(MemoryConfigurationType.ADJACENCY, {
                'memory_size': 0
        }),
    }
    model = Model(g, ModelConfiguration(cfg))

    constants = {
        'link_addition_p': 0.5,
        'link_removal_p': n * 0.001,

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
        'A': 0
    }

    def sample_attention_weighted(graph):
        A = model.get_state('A')
        if max(A) == 0:
            return []
        else:
            probs = []
            factor = 1.0/sum(A)
            for a in A:
                probs.append(a * factor)
            return np.random.choice(graph.nodes, size=1, replace=False, p=probs)

    def update(constants):
        information = model.get_state('I')
        attention = model.get_state('A')
        opinion = model.get_state('O')

        if max(attention) == 0:
            agent = 0
        else:
            factor = 1.0/sum(attention)
            probs = []
            for a in attention:
                probs.append(a * factor)
            agent = np.random.choice(list(range(constants['N'])), 1, replace=False, p=probs)

        if agent != 0:
            neighbors = model.get_neighbors(agent[0])
            if len(neighbors) > 0:
                partner = np.random.choice(neighbors)
            else:
                partner = 0
        else:
            partner = 0

        if partner != 0 and agent != 0:
            I1 = information[agent]; A1 = attention[agent]; O1 = opinion[agent]
            I2 = information[partner]; A2 = attention[partner]; O2 = opinion[partner]

            if abs(O1 - O2) < constants['deffuant_c']:
                information[agent] = information_update(I1,I2,A1,A2,constants['persuasion'],constants['r_min'],True)
                information[partner] = information_update(I2,I1,A2,A1,constants['persuasion'],constants['r_min'],True)

            if abs(O1 - O2) < constants['deffuant_c']:
                attention[agent]=attention[agent]+constants['delta_attention']*(2*constants['attention_star']-attention[agent])
                attention[partner]=attention[partner]+constants['delta_attention']*(2*constants['attention_star']-attention[partner])

        information = information+np.random.normal(0,constants['sd_noise_information'],constants['N'])
        attention = attention-2*constants['delta_attention']*attention/constants['N']   # correction 2 times if interaction
        opinion = stoch_cusp(constants['N'],opinion,attention+constants['min_attention'],information,constants['s_O'], constants['maxwell_convention'])

        return {'O': opinion, 'I': information, 'A': attention}

    def activism():
        information = model.get_state('I')
        attention = model.get_state('A')
        opinion = model.get_state('O')

        m = np.zeros(constants['N'])
        m[np.arange(1, constants['N'], constants['N']/3)[1:].astype(int)-1] = 1

        information_activists=-.5
        attention_activists=1
        opinion_activists=-.5

        information[m == 1] = information_activists
        attention[m == 1] = attention_activists
        opinion[m == 1] = opinion_activists

        return {'O': opinion, 'I': information, 'A': attention}

    def network_update(nodes, constants):
        node = nodes[0]
        update = {}
        adj = model.get_adjacency()

        # # 50% chance for an agent to create a new connection
        if np.random.random_sample() <= constants['link_addition_p']:
            # Pick a new link that does not exist yet for this agent
            new_link = np.random.choice(np.where(adj[node] == 0)[0])
            update['add'] = [new_link]

        if np.random.random_sample() <= constants['link_removal_p']:
            # Pick a link that does exists yet for this agent
            neighbors = np.where(adj[node] == 1)[0]
            if len(neighbors) >= 1:
                link = np.random.choice(neighbors)
                update['remove'] = [link]
        return {
            'edge_change': {
                node: update
            }
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

    activism_update = Update(activism)
    model.add_scheme(Scheme(lambda graph: graph.nodes, {'args': {'graph': model.graph}, 'lower_bound': 300, 'upper_bound': 301, 'updates': [activism_update]}))

    model.set_initial_state(initial_state, {'constants': model.constants})

    output = model.simulate(1000)

    visualization_config = {
        'layout': nx.drawing.layout.spring_layout,
        'plot_interval': 10,
        'plot_variable': 'O',
        'variable_limits': {
            'A': [0, 1],
            'O': [-1, 1],
            'I': [-1, 1]
        },
        'cmin': -1,
        'cmax': 1,
        'color_scale': 'RdBu',
        'show_plot': False,
        'plot_output': '../animations/HIOM_dynamic.gif',
        'plot_title': 'HIERARCHICAL ISING OPINION MODEL',
    }

    model.configure_visualization(visualization_config, output)
    model.visualize('animation')
