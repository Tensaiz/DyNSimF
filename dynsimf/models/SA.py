# import multiprocessing as mp
import copy
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class SAConfiguration(object):
    """
    Configuration for Sensitivity Analysis
    TODO: Validate attributes
    """
    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)
        self.validate()

    def validate(self):
        pass


class SensitivityAnalysis(object):

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.states = list(self.model.state_map.keys())

    def get_saltelli_params(self):
        problem = {
            'num_vars': len(self.config.bounds.keys()),
            'names': [var for var in self.config.bounds.keys()],
            'bounds': [
                [lower, upper] for _, (lower, upper) in self.config.bounds.items()
            ]
        }
        return problem, saltelli.sample(problem, self.config.n, calc_second_order=self.config.second_order)

    def analyze_sensitivity(self):

        problem, param_values = self.get_saltelli_params()

        print('Running Simulation...')
        outputs = []
        # Optimize by using parallel processing to run multiple simulations at once
        for i in range(len(param_values)):
            print('Running simulation ' + str(i + 1) + '/' + str(len(param_values)))
            outputs.append(self.run_model(param_values[i], problem['names']))

        print('Parsing outputs...')
        out = self.parse(outputs)
        print(out)
        print('Running sensitivity analysis...')
        return self.analyze_output(problem, out)

    def run_model(self, params, names):
        self.set_model(params, names)

        if self.config.algorithm_input == 'network':
            self.model.simulate(self.config.iterations, show_tqdm=False)
            return copy.deepcopy(self.model.graph)
        else:
            return self.model.simulate(self.config.iterations, show_tqdm=False)

    def set_model(self, params, names):
        for i, name in enumerate(names):
            self.model.constants[name] = params[i]
        self.model.set_initial_state(self.config.initial_state, self.config.initial_args)

    def parse(self, outputs):
        mapping = {
            'mean': np.mean,
            'variance': np.var,
            'min': np.min,
            'max': np.max
        }
        if self.config.algorithm in list(mapping.keys()):
            return self.state_reduce(outputs, mapping[self.config.algorithm])
        else:
            return self.custom_reduce(outputs)

    def custom_reduce(self, outputs):
        if self.config.output_type == 'reduce':
            res = np.array([])
            for output in outputs:
                res = np.append(res, self.config.algorithm(output, **self.config.algorithm_args))
        else:
            return self.state_reduce(outputs, self.config.algorithm, self.config.algorithm_args)
        return res

    def state_reduce(self, outputs, fun, args=None):
        arguments = args if args else {}
        res = self.get_state_dict()
        for output in outputs:
            for state in self.states:
                res[state] = np.append(res[state], fun(output[-1][:, self.model.state_map[state]], **arguments))
        return res

    def get_state_dict(self):
        return {
            var: np.array([]) for var in self.states
        }

    def analyze_output(self, problem, output):
        if isinstance(output, dict):
            # Perform the sobol analysis seperately for every status
            return {
                var: sobol.analyze(problem, output[var]) for var in self.states
            }
        elif isinstance(output, np.ndarray):
            return sobol.analyze(problem, output)
