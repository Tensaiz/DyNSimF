import networkx as nx
import numpy as np

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration
from dynsimf.models.components.Scheme import Scheme
from dynsimf.models.components.Update import Update


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
    }
    model = Model(g, ModelConfiguration(cfg))

    constants = {
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

    def update(constants):
        information = model.get_state('I')
        attention = model.get_state('A')
        opinion = model.get_state('O')

        if max(attention) == 0:
            agent = -1
        else:
            factor = 1.0/sum(attention)
            probs = []
            for a in attention:
                probs.append(a * factor)
            agent = np.random.choice(list(range(constants['N'])), 1, replace=False, p=probs)

        if agent != -1:
            partner = np.random.choice(model.get_neighbors(agent[0]))
        else:
            partner = -1

        if partner != -1 and agent != -1:
            I1 = information[agent]; A1 = attention[agent]; O1 = opinion[agent]
            I2 = information[partner]; A2 = attention[partner]; O2 = opinion[partner]

        if partner != -1 and agent != -1:
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

    # Model definition
    model.constants = constants
    model.set_states(['I', 'A', 'O'])

    model.add_update(update, {'constants': model.constants})

    activism_update = Update(activism)
    model.add_scheme(Scheme(lambda graph: graph.nodes, {'args': {'graph': model.graph}, 'lower_bound': 300, 'upper_bound': 301, 'updates': [activism_update]}))


    model.set_initial_state(initial_state, {'constants': model.constants})

    output = model.simulate(15000)

    visualization_config = {
        'layout': nx.drawing.layout.spring_layout,
        'plot_interval': 500,
        'plot_variable': 'O',
        'variable_limits': {
            'A': [0, 1.5],
            'O': [-1, 1],
            'I': [-1, 1]
        },
        'cmin': -1,
        'cmax': 1,
        'color_scale': 'RdBu',
        'show_plot': True,
        # 'plot_output': '../animations/HIOM.gif',
        'plot_title': 'HIERARCHICAL ISING OPINION MODEL',
    }

    model.configure_visualization(visualization_config, output)
    model.visualize('animation')
