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

    :var dict bounds: a dictionary with
        (str) constant (key): tuple(lower_bound, upper_bound) for indicating the bounds to sample the constants from
    :var int iterations: The amount of iterations to run the model for one simulation
    :var dict initial_state: The initial state of the model per simulation
    :var dict initial_args: Initial arguments to pass to initial state function if provided
    :var int n: The amount of samples to generate
    :var bool second_order: Whether to set calc_second_order to true when Using Saltelli sample from SALib

    :var str algorithm_input: When set to 'network', the networkx graph will be given as output,
    if set to anything else the states will be outputted per simulation 
    :var function algorithm: The algorithm or function to use on the state values, can be any function that results in one value for each node per state
    (e.g: mean, variance, max, or any other)
    :var dict algorithm_args: A dictionary with custom input params for the provided algorithm function
    :var str output_type: If the output_type is set to 'reduce', 
    the algorithm that is provided will be applied be applied on every simulation run and the output is appended to an array
    This allows custom algorithms to transform the output in whatever shape is required,
    Otherwise the algorithm takes as default input a column that contains the values for all the nodes for one state
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
    '''
    
    '''
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.states = list(self.model.state_map.keys())

    def get_saltelli_params(self):
        '''
        Use saltelli sample from SALib to generate samples to use for the simulation runs
        And create the exact problem dictionary that SALib requires.
        '''
        problem = {
            'num_vars': len(self.config.bounds.keys()),
            'names': [var for var in self.config.bounds.keys()],
            'bounds': [
                [lower, upper] for _, (lower, upper) in self.config.bounds.items()
            ]
        }
        return problem, saltelli.sample(problem, self.config.n, calc_second_order=self.config.second_order)

    def analyze_sensitivity(self):
        '''
        The main entry point to analyze the sensitivity of a model to its constants

        First the constants are sampled using the saltelli sample function,
        then the model is simulation for all the samples generated.
        Each model simulation takes 'iterations' iterations from the config.
        After the simulation, the output is reduced so that each model run, results in one value per state.
        By default only the last iteration of a simulation is considered.
        This is done using the provided algorithm in the config.

        Finally, the outputs are analyzed using sobol.analyze from SALib.
        This results in an output dictionary containing the output of sobol.analyze as value for each state (key)
        '''
        problem, param_values = self.get_saltelli_params()
        print('Running Simulation...')
        outputs = []
        # Optimize by using parallel processing to run multiple simulations at once
        for i in range(len(param_values)):
            print('Running simulation ' + str(i + 1) + '/' + str(len(param_values)))
            outputs.append(self.run_model(param_values[i], problem['names']))
        print('Parsing outputs...')
        out = self.parse(outputs)
        print('Running sensitivity analysis...')
        return self.analyze_output(problem, out)

    def run_model(self, params, names):
        '''
        Simulate the model for 'iterations' iterations and either return the networkx graph as output, or the complete simulation output (states, etc) 
        '''
        self.set_model(params, names)

        if self.config.algorithm_input == 'network':
            self.model.simulate(self.config.iterations, show_tqdm=False)
            return copy.deepcopy(self.model.graph)
        else:
            return self.model.simulate(self.config.iterations, show_tqdm=False)

    def set_model(self, params, names):
        '''
        Reset the model and set the new sampled params
        '''
        self.model.reset()
        for i, name in enumerate(names):
            self.model.constants[name] = params[i]
        self.model.set_initial_state(self.config.initial_state, self.config.initial_args)

    def parse(self, outputs):
        '''
        Flatten the output of all the iterations per simulation runs, for each node and their states
        Then apply an algorithm on the runs/nodes/states to get one value for each simulation run that can be analyzed
        '''
        transformed_outputs = self.flatten_output(outputs)
        return self.apply_algorithm(transformed_outputs)

    def flatten_output(self, outputs):
        '''
        Transform the output of the model run to a list that contains the values of the last iteration and the states in order
        The output is a list of 2d arrays that contain columns for states and rows for nodes where each 2d array is one model run
        '''
        flattened_output = []
        for model_run in outputs:
            flattened_output.append(model_run['states'][self.config.iterations])
        return flattened_output

    def apply_algorithm(self, outputs):
        ''' 
        Apply a known algorithm, or a custom algorithm on the output of the simulation runs,
        this will reduce the values to one value per state per simulation.
        '''
        mapping = {
            'mean': np.mean,
            'variance': np.var,
            'min': np.min,
            'max': np.max
        }
        if self.config.algorithm in list(mapping.keys()):
            return self.state_reduce(outputs, mapping[self.config.algorithm])
        elif self.config.output_type == 'reduce':
            return self.custom_reduce(outputs)
        else:
            return self.state_reduce(outputs, self.config.algorithm, self.config.algorithm_args)

    def custom_reduce(self, outputs):
        '''
        If the output type in the config is set to 'reduce', 
        the algorithm that is provided will be applied be applied on every simulation run and the output is appended to an array
        This allows custom algorithms to transform the output in whatever shape is required
        '''
        res = np.array([])
        for output in outputs:
            res = np.append(res, self.config.algorithm(output, **self.config.algorithm_args))
        return res

    def state_reduce(self, outputs, fun, args=None):
        '''
        Create a dictionary that holds states as keys 
        and reduces the values across all nodes to a single value using a provided algorithm
        (e.g: average value for all nodes per state)
        '''
        arguments = args if args else {}
        res = self.get_state_dict()
        for output in outputs:
            for state in self.states:
                res[state] = np.append(res[state], fun(output[:, self.model.state_map[state]], **arguments))
        return res

    def get_state_dict(self):
        '''
        Return a dictionary with the states as keys, and an empty numpy array as value
        '''
        return {
            var: np.array([]) for var in self.states
        }

    def analyze_output(self, problem, output):
        '''
        Analyze the output after reducing the values using the sobol.analyze function from SALib
        The returned value is either a dictionary with an analysis per state, or a single analysis object if there is only one input list
        '''
        if isinstance(output, dict):
            # Perform the sobol analysis seperately for every status
            return {
                var: sobol.analyze(problem, output[var]) for var in self.states
            }
        elif isinstance(output, np.ndarray):
            return sobol.analyze(problem, output)
