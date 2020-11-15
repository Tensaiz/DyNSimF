# import multiprocessing as mp
import copy
import numpy as np
import types

from dynsimf.models.helpers.ConfigValidator import ConfigValidator

from SALib.sample import saltelli
from SALib.analyze import sobol

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class SAConfiguration(object):
    """
    Configuration for Sensitivity Analysis
    TODO: Validate attributes
    """
    def __init__(self, config):
        self.set_config(config)
        self.validate()

    def set_config(self, config):
        self.config = config
        self.bounds = config['bounds'] if 'bounds' in config else None
        self.iterations = config['iterations'] if 'iterations' in config else None
        self.initial_state = config['initial_state'] if 'initial_state' in config else None
        self.initial_args = config['initial_args'] if 'initial_args' in config else {}
        self.n = config['n'] if 'n' in config else None
        self.second_order = config['second_order'] if 'second_order' in config else False
        self.algorithm_input = config['algorithm_input'] if 'algorithm_input' in config else None
        self.algorithm = config['algorithm'] if 'algorithm' in config else None
        self.output_type = config['output_type'] if 'output_type' in config else None
        self.algorithm_args = config['algorithm_args'] if 'algorithm_args' in config else {}

    def validate(self):
        ConfigValidator.validate('bounds', self.bounds, dict)
        ConfigValidator.validate('iterations', self.iterations, int)
        ConfigValidator.validate('initial_state', self.initial_state, dict)
        ConfigValidator.validate('initial_args', self.initial_args, dict)
        ConfigValidator.validate('n', self.n, int)
        ConfigValidator.validate('second_order', self.second_order, bool)
        ConfigValidator.validate('algorithm_input', self.algorithm_input, str)
        ConfigValidator.validate('algorithm', self.algorithm, types.FunctionType)
        ConfigValidator.validate('output_type', self.output_type, str)
        ConfigValidator.validate('algorithm_args', self.algorithm_args, dict)

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
