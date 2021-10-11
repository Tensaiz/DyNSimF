from dynsimf.models.Model import Model
from dynsimf.models.tools.SA import SensitivityAnalysis
from dynsimf.models.tools.SA import SAConfiguration

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from SALib.test_functions import Ishigami

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


if __name__ == "__main__":
    g = nx.random_geometric_graph(1, 1)
    model = Model(g)

    constants = {
        'x1': 0,
        'x2': 0,
        'x3': 0
    }

    initial_state = {
        'ishigami': 0
    }

    def update(constants):
        ishigami_params = np.array([list(constants.values())])
        return {'ishigami': Ishigami.evaluate(ishigami_params)}

    # Model definition
    model.constants = constants
    model.set_states(['ishigami'])
    model.add_update(update, {'constants': model.constants})

    cfg = SAConfiguration(
        {
            'bounds': {'x1': (-3.14159265359, 3.14159265359), 'x2': (-3.14159265359, 3.14159265359), 'x3': (-3.14159265359, 3.14159265359)},
            'iterations': 1,
            'initial_state': initial_state,
            'initial_args': {'constants': model.constants},
            'n': 1024,
            'second_order': True,

            'algorithm_input': 'states',
            'algorithm': lambda x: x,
            'output_type': '',
            'algorithm_args': {},
        }
    )
    sa = SensitivityAnalysis(cfg, model)
    analysis = sa.analyze_sensitivity()
    print(analysis)
    print("x1-x2:", analysis['ishigami']['S2'][0,1])
    print("x1-x3:", analysis['ishigami']['S2'][0,2])
    print("x2-x3:", analysis['ishigami']['S2'][1,2])
    analysis['ishigami'].plot()
    plt.show()
