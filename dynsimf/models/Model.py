from abc import ABCMeta
import tqdm
import copy
import numpy as np
import networkx as nx

from dynsimf.models.components.Memory import MemoryConfiguration
from dynsimf.models.components.Memory import MemoryConfigurationType
from dynsimf.models.components.Update import Update
from dynsimf.models.components.Update import UpdateType
from dynsimf.models.components.Update import UpdateConfiguration
from dynsimf.models.components.Scheme import Scheme
from dynsimf.models.tools.Visualizer import VisualizationConfiguration
from dynsimf.models.tools.Visualizer import Visualizer

from typing import List

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class ConfigurationException(Exception):
    """Configuration Exception"""


class ModelConfiguration(object):
    """
    Configuration for the model
    """
    def __init__(self, cfg=None):
        cfg = cfg if cfg else {}
        cfg_keys = list(cfg.keys())
        self.utility = False if 'utility' not in cfg_keys else cfg['utility']
        self.state_memory_config = \
            MemoryConfiguration(MemoryConfigurationType.STATE, {
                'memory_size': 0
            }) \
            if 'state_memory_config' not in cfg_keys \
            else cfg['state_memory_config']
        self.utility_memory_config = MemoryConfiguration(MemoryConfigurationType.UTILITY) \
            if 'utility_memory_config' not in cfg_keys \
            else cfg['utility_memory_config']
        self.adjacency_memory_config = \
            MemoryConfiguration(MemoryConfigurationType.ADJACENCY, {
                'memory_size': -1
            }) \
            if 'adjacency_memory_config' not in cfg_keys \
            else cfg['adjacency_memory_config']
        self.edge_values_memory_config = \
            MemoryConfiguration(MemoryConfigurationType.EDGE_VALUES, {
                'memory_size': -1
            }) \
            if 'adjacency_memory_config' not in cfg_keys \
            else cfg['adjacency_memory_config']

class Model(object, metaclass=ABCMeta):
    """
    Partial Abstract Class defining a model
    """
    def __init__(self, graph, config=None, seed=None):
        self.graph = graph
        self.config = config if config else ModelConfiguration()
        self.clear()
        self.init()
        np.random.seed(seed)

    @property
    def constants(self):
        return self.__constants

    @constants.setter
    def constants(self, constants):
        self.__constants = constants

    @property
    def nodes(self):
        return list(self.graph.nodes())

    def init(self):
        self.graph_changed = False
        self.adjacency = nx.convert_matrix.to_numpy_array(self.graph)
        self.new_adjacency = self.adjacency[:].copy()
        self.graph = nx.convert_matrix.from_numpy_array(self.adjacency)
        self.new_graph = copy.deepcopy(self.graph)
        if self.config.utility:
            self.initialize_utility()

    def add_property_function(self, fun):
        self.property_functions.append(fun)

    def set_states(self, state_names):
        self.node_states = np.zeros((len(self.graph.nodes()), len(state_names)))
        self.new_node_states = self.node_states[:].copy()
        self.state_names = state_names
        for i, state in enumerate(state_names):
            self.state_map[state] = i

    def set_edge_values(self, edge_values_names):
        self.edge_values = np.zeros((len(edge_values_names), len(self.graph.nodes()), len(self.graph.nodes())))
        self.new_edge_values = self.edge_values[:].copy()
        self.edge_values_names = edge_values_names
        for i, edge_values_name in enumerate(edge_values_names):
            self.edge_values_map[edge_values_name] = i

    def set_initial_state(self, initial_state, args=None):
        arguments = args if args else {}
        for state in initial_state.keys():
            val = initial_state[state]
            if hasattr(val, '__call__'):
                self.node_states[:, self.state_map[state]] = val(**arguments)
            else:
                self.node_states[:, self.state_map[state]] = val
        self.new_node_states = self.node_states[:].copy()

    def set_initial_edge_values(self, initial_edge_values, args=None):
        arguments = args if args else {}
        for edge_value_name in initial_edge_values.keys():
            val = initial_edge_values[edge_value_name]
            if hasattr(val, '__call__'):
                self.edge_values[self.edge_values_map[edge_value_name], :, :] = val(**arguments)
            else:
                self.edge_values[self.edge_values_map[edge_value_name], :, :] = val
        self.new_edge_values = self.edge_values[:].copy()

    def initialize_utility(self):
        n_nodes = len(self.graph.nodes())
        self.edge_utility = np.zeros((n_nodes, n_nodes))
        self.new_edge_utility = self.edge_utility[:].copy()

    def get_utility(self):
        return self.edge_utility

    def get_nodes_utility(self, nodes):
        return self.edge_utility[nodes]

    def set_initial_utility(self, init_function, params=None):
        params = params if params else {}
        self.edge_utility = init_function(*params)
        self.new_edge_utility = self.edge_utility[:].copy()

    def get_state_index(self, state):
        return self.state_map[state]

    def get_state(self, state):
        return self.node_states[:, self.state_map[state]]

    def get_node_states(self, node):
        return self.node_states[node]

    def get_node_state(self, node, state):
        return self.node_states[node, self.state_map[state]]

    def get_nodes_state(self, nodes, state):
        return self.node_states[nodes, self.state_map[state]]

    def get_nodes_states(self):
        return self.node_states

    def get_previous_nodes_states(self, n):
        """
        Get all the nodes' states from the n'th previous saved iteration
        """
        available_iterations = list(self.simulation_output['states'].keys())
        return self.simulation_output['states'][available_iterations[-n - 1]]

    def get_previous_nodes_utility(self, n):
        """
        Get all the nodes' utility from the n'th previous saved iteration
        """
        available_iterations = list(self.simulation_output['utility'].keys())
        return self.simulation_output['utility'][available_iterations[-n - 1]]

    def get_previous_nodes_adjacency(self, n):
        """
        Get all the adjacency matrix from the n'th previous saved iteration
        """
        available_iterations = list(self.simulation_output['adjacency'].keys())
        return self.simulation_output['adjacency'][available_iterations[-n - 1]]

    def add_update(self, fun, args=None, condition=None, get_nodes=False, update_type=None):
        arguments = args if args else {}
        if condition:
            self.set_conditions_state_indices(condition)
        update_type = update_type if update_type else UpdateType.STATE
        update = self.create_update((fun, arguments, condition, get_nodes, update_type))
        self.schemes[0].add_update(update)

    def set_conditions_state_indices(self, condition):
        current_condition = condition
        while current_condition:
            if current_condition.get_state():
                current_condition.set_state_index(self.state_map[current_condition.get_state()])
            current_condition = current_condition.chained_condition

    def add_state_update(self, fun, args=None, condition=None, get_nodes=False):
        self.add_update(fun, args, condition, get_nodes, UpdateType.STATE)

    def add_utility_update(self, fun, args=None, condition=None, get_nodes=False):
        if self.config.utility == False:
            raise ValueError('Utility has not been set to true in config')
        self.add_update(fun, args, condition, get_nodes, UpdateType.UTILITY)

    def add_network_update(self, fun, args=None, condition=None, get_nodes=False):
        self.add_update(fun, args, condition, get_nodes, UpdateType.NETWORK)

    def add_edge_values_update(self, fun, args=None, condition=None, get_nodes=False):
        self.add_update(fun, args, condition, get_nodes, UpdateType.EDGE_VALUES)

    def create_update(self, update_content):
        fun, arguments, condition, get_nodes, update_type = update_content
        cfg_options = {
            'arguments': arguments,
            'condition': condition,
            'get_nodes': get_nodes,
            'update_type': update_type
        }
        update_cfg = UpdateConfiguration(cfg_options)
        return Update(fun, update_cfg)

    def add_scheme(self, scheme):
        self.set_scheme_update_condition_state_indices(scheme)
        self.schemes.append(scheme)

    def set_scheme_update_condition_state_indices(self, scheme):
        if scheme.updates:
            for update in scheme.updates:
                if update.condition:
                    self.set_conditions_state_indices(update.condition)

    def get_adjacency(self):
        return self.adjacency

    def get_nodes_adjacency(self, nodes):
        return self.adjacency[nodes]

    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))

    def get_neighbors_neighbors_adjacency_matrix(self):
        adj_neighbors_neighbors = self.adjacency @ self.adjacency
        # Doesn't matter how many connections are shared
        adj_neighbors_neighbors[adj_neighbors_neighbors > 0] = 1
        # Remove direct neighbors
        adj_neighbors_neighbors = adj_neighbors_neighbors - self.adjacency
        # Clean matrix
        adj_neighbors_neighbors[adj_neighbors_neighbors < 0] = 0
        # Can't have connection to yourself
        np.fill_diagonal(adj_neighbors_neighbors, 0)
        return adj_neighbors_neighbors

    def get_neighbors_neighbors(self, node):
        neighbors_neighbors_matrix = self.get_neighbors_neighbors_adjacency_matrix()
        return np.array(self.graph.nodes)[np.where(neighbors_neighbors_matrix[node] > 0)[0]]

    def simulate(self, n, show_tqdm=True):
        self.simulation_output = {
            'states': {},
            'adjacency': {},
            'utility': {},
            'edge_values': {}
        }
        self.prepare_output(n)
        self.simulation_steps(n, show_tqdm)
        return self.simulation_output

    def prepare_output(self, n):
        n_nodes = len(self.graph.nodes)
        if self.config.state_memory_config.save_disk:
            with open(self.config.state_memory_config.path, 'w') as f:
                f.write(self.config.state_memory_config.get_description_string(n, n_nodes, len(self.state_names)))
        if self.config.adjacency_memory_config.save_disk:
            with open(self.config.adjacency_memory_config.path, 'w') as f:
                f.write(self.config.adjacency_memory_config.get_description_string(n, n_nodes, n_nodes))
        if self.config.utility_memory_config.save_disk:
            with open(self.config.utility_memory_config.path, 'w') as f:
                f.write(self.config.utility_memory_config.get_description_string(n, n_nodes, n_nodes))

    def simulation_steps(self, n, show_tqdm):
        self.store_simulation_step()
        if show_tqdm:
            for _ in tqdm.tqdm(range(0, n)):
                self.simulation_step()
        else:
            for _ in range(0, n):
                self.simulation_step()

    def simulation_step(self):
        self.iteration()
        self.write_simulation_step()
        self.store_simulation_step()

    def write_simulation_step(self):
        n_nodes = len(self.graph.nodes)
        if self.config.state_memory_config.save_disk and self.current_iteration % self.config.state_memory_config.save_interval == 0:
            self.write_states_iteration(n_nodes)
        if self.config.utility_memory_config.save_disk and self.current_iteration % self.config.utility_memory_config.save_interval == 0:
            self.write_utility_iteration(n_nodes)
        if self.config.adjacency_memory_config.save_disk and self.current_iteration % self.config.adjacency_memory_config.save_interval == 0:
            self.write_adjacency_iteration(n_nodes)

    def write_states_iteration(self, n_nodes):
        with open(self.config.state_memory_config.path, 'a') as f:
            f.write('# Iteration {0} - ({1}, {2})\n'.format(self.current_iteration, n_nodes, len(self.state_names)))
            np.savetxt(f, self.node_states)

    def write_utility_iteration(self, n_nodes):
        with open(self.config.utility_memory_config.path, 'a') as f:
            f.write('# Iteration {0} - ({1}, {2})\n'.format(self.current_iteration, n_nodes, n_nodes))
            np.savetxt(f, self.edge_utility)

    def write_edge_values_iteration(self, n_nodes):
        pass

    def write_adjacency_iteration(self, n_nodes):
        with open(self.config.adjacency_memory_config.path, 'a') as f:
            f.write('# Iteration {0} - ({1}, {2})\n'.format(self.current_iteration, n_nodes, n_nodes))
            np.savetxt(f, self.adjacency)

    def store_simulation_step(self):
        self.store_states_iteration()
        self.store_utility_iteration()
        self.store_adjacency_iteration()
        self.store_edge_values_iteration()

    def store_states_iteration(self):
        if self.config.state_memory_config.memory_size != -1 and self.current_iteration % self.config.state_memory_config.memory_interval == 0:
            self.simulation_output['states'][self.current_iteration] = copy.deepcopy(self.node_states)
        if self.config.state_memory_config.memory_size > 0 and len(self.simulation_output) > self.config.state_memory_config.memory_size:
            self.simulation_output['states'] = {}

    def store_edge_values_iteration(self):
        if self.config.edge_values_memory_config.memory_size != -1 and self.current_iteration % self.config.edge_values_memory_config.memory_interval == 0:
            self.simulation_output['edge_values'][self.current_iteration] = copy.deepcopy(self.edge_values)
        if self.config.edge_values_memory_config.memory_size > 0 and len(self.simulation_output) > self.config.edge_values_memory_config.memory_size:
            self.simulation_output['edge_values'] = {}

    def store_utility_iteration(self):
        if self.config.utility_memory_config.memory_size != -1 and self.current_iteration % self.config.utility_memory_config.memory_interval == 0:
            self.simulation_output['utility'][self.current_iteration] = copy.deepcopy(self.edge_utility)
        if self.config.utility_memory_config.memory_size > 0 and len(self.simulation_output) > self.config.utility_memory_config.memory_size:
            self.simulation_output['utility'] = {}

    def store_adjacency_iteration(self):
        if self.config.adjacency_memory_config.memory_size != -1 and self.current_iteration % self.config.adjacency_memory_config.memory_interval == 0:
            self.simulation_output['adjacency'][self.current_iteration] = copy.deepcopy(self.adjacency)
        if self.config.adjacency_memory_config.memory_size > 0 and len(self.simulation_output) > self.config.adjacency_memory_config.memory_size:
            self.simulation_output['adjacency'] = {}

    def iteration(self):
        self.iteration_calculation()
        self.iteration_assignment()
        self.calculate_properties()
        self.prepare_next_iteration()

    def iteration_calculation(self):
        # For every scheme
        for scheme in self.schemes:
            if self.inactive_scheme(scheme):
                continue
            scheme_nodes = np.array(scheme.sample())
            # For all the updates in the scheme
            for update in scheme.updates:
                update_nodes = self.valid_update_condition_nodes(update, scheme_nodes)
                if (len(update_nodes) == 0):
                    continue
                if update.get_nodes:
                    updatables = update.execute(update_nodes)
                else:
                    updatables = update.execute()
                self.assign_update(update, update_nodes, updatables)

    def assign_update(self, update, update_nodes, updatables):
        if update.update_type == UpdateType.STATE:
            self.update_state(update_nodes, updatables)
        elif update.update_type == UpdateType.UTILITY:
            self.update_utility(update_nodes, updatables)
        elif update.update_type == UpdateType.NETWORK:
            self.update_network(update_nodes, updatables)
        elif update.update_type == UpdateType.EDGE_VALUES:
            self.update_edge_values(update_nodes, updatables)

    def update_state(self, nodes, updatables):
        for state, update_output in updatables.items():
            if isinstance(update_output, list) or isinstance(update_output, np.ndarray):
                self.new_node_states[nodes, self.state_map[state]] = update_output
            elif isinstance(update_output, dict):
                # Add a 2d array implementation instead of for loop
                for node, values in update_output.items():
                    self.new_node_states[node, self.state_map[state]] = values

    def update_utility(self, update_nodes, updatables):
        if isinstance(updatables, np.ndarray):
            self.new_edge_utility[update_nodes] = updatables
        elif isinstance(updatables, dict):
            self.update_utility_specific_edges(updatables)

    def update_utility_specific_edges(self, utility_list):
        for (origin, neighbor, utility) in utility_list:
            self.new_edge_utility[origin, neighbor] = utility

    def update_edge_values(self, update_nodes, updatables):
        for edge_values_name, update_output in updatables.items():
            self.new_edge_values[self.edge_values_map[edge_values_name], update_nodes] = update_output

    def update_network(self, update_nodes, updatables):
        for network_update_type, change in updatables.items():
            self.assign_network_operation(network_update_type, change, update_nodes)

    def assign_network_operation(self, network_update_type, change, update_nodes):
        network_update_type_to_function = {
            'remove': self.network_nodes_remove,
            'add': self.network_nodes_add,
            'edge_change': self.network_edges_change
        }
        network_update_type_to_function[network_update_type](change, update_nodes)

    def network_nodes_remove(self, removable_nodes, update_nodes):
        self.new_node_states = np.delete(self.new_node_states, removable_nodes, axis=0)
        # self.delete_rows_columns('new_edge_utility', removable_nodes)
        self.delete_rows_columns('new_adjacency', removable_nodes)

    def delete_rows_columns(self, var, removables):
        setattr(self, var, np.delete(getattr(self, var), removables, axis=0))
        setattr(self, var, np.delete(getattr(self, var), removables, axis=1))

    def network_nodes_add(self, new_nodes, _):
        """
        Add a list of new node dictionaries to the model
        """
        node_index = len(self.new_node_states)
        for node in new_nodes:
            self.initialize_new_node()
            self.handle_node_initialization(node_index, node)
            node_index += 1

    def initialize_new_node(self):
        """
        Add a new row of 0s to the new adjacency, new edge utilities and new node states matrices
        Also add a new column of 0s to the new adjacency and new edge utilities
        """
        self.new_adjacency = np.vstack([self.new_adjacency, np.zeros(len(self.new_adjacency))])
        self.new_adjacency = np.append(self.new_adjacency, np.zeros((len(self.new_adjacency), 1)), axis=1)
        # self.new_edge_utility = np.vstack([self.new_edge_utility, np.zeros(len(self.new_adjacency))])
        # self.new_edge_utility = np.append(self.new_edge_utility, np.zeros((len(self.new_adjacency), 1)), axis=1)
        self.new_node_states = np.vstack([self.new_node_states, np.zeros(len(self.state_names))])
        self.graph_changed = True

    def handle_node_initialization(self, index, node):
        """
        :param index int: The index of the node in the graph
        :param node dict: a node dictionary of the form:
            key (str): 'neighbors', value (list[tuple]): (neighbor_index, edge_values_name values_in, values_out)
            key (str): 'states', value (dict): { 'state_name' (str) : state_value (number) }
        Initialize a node by setting the utility of the neighbors and setting the node states
        """
        self.set_node_neighbor_values(index, node['neighbors'])
        self.set_new_node_states(index, node['states'])

    def set_new_node_states(self, index, states):
        for state, value in states.items():
            self.new_node_states[index, self.state_map[state]] = value

    def network_edges_change(self, change, update_nodes):
        for origin, node_changes in change.items():
            for adjacency_change_type, neighbors in node_changes.items():
                if adjacency_change_type == 'overwrite':
                    self.handle_adjacency_node_overwrite(origin, neighbors)
                elif adjacency_change_type == 'add':
                    self.handle_adjacency_node_add(origin, neighbors)
                elif adjacency_change_type == 'remove':
                    self.handle_adjacency_node_remove(origin, neighbors)
    def handle_adjacency_node_overwrite(self, origin, neighbors):
        """
        neighbors variable format: (neighbor_index, edge_variable_name, origin_to_neighbor_val, neighbor_to_origin_val)
        """
        neighbor_indices = self.neighbor_update_to_var('indices', neighbors)
        # Clear node current neighbors
        self.new_adjacency[origin] = 0
        # Clear all neighbors connected to origin node
        self.new_adjacency[:, origin] = 0
        # Set node -> neighbors adjacency
        self.new_adjacency[origin, neighbor_indices] = 1
        # Set neighbors -> node adjacency
        self.new_adjacency[neighbor_indices, origin] = 1
        # Clear all ingoing and outgoing edge values
        for edge_value_name in self.edge_values_names:
            self.new_edge_values[self.edge_values_map[edge_value_name], origin] = 0
            self.new_edge_values[self.edge_values_map[edge_value_name], :, origin] = 0
        # Set given utility
        self.set_node_neighbor_values(origin, neighbors)
        self.graph_changed = True

    def handle_adjacency_node_add(self, origin, neighbors):
        neighbor_indices = self.neighbor_update_to_var('indices', neighbors)
        self.new_adjacency[origin, neighbor_indices] = 1
        self.new_adjacency[neighbor_indices, origin] = 1
        self.set_node_neighbor_values(origin, neighbors)
        self.graph_changed = True

    def set_node_neighbor_values(self, origin, neighbors):
        neighbor_indices = self.neighbor_update_to_var('indices', neighbors)
        edge_value_names = self.neighbor_update_to_var('names', neighbors)
        neighbor_ingoing_values = self.neighbor_update_to_var('ingoing_values', neighbors)
        neighbor_outgoing_values = self.neighbor_update_to_var('outgoing_values', neighbors)
        self.new_edge_values[edge_value_names, origin, neighbor_indices] = neighbor_outgoing_values
        self.new_edge_values[edge_value_names, neighbor_indices, origin] = neighbor_ingoing_values

    def handle_adjacency_node_remove(self, origin, neighbors):
        self.new_adjacency[origin, neighbors] = 0
        self.new_adjacency[neighbors, origin] = 0
        for edge_value_name in self.edge_values_names:
            self.new_edge_values[self.edge_values_map[edge_value_name], origin, neighbors] = 0
            self.new_edge_values[self.edge_values_map[edge_value_name], neighbors, origin] = 0
        self.graph_changed = True

    def neighbor_update_to_var(self, var_type, neighbors):
        """
        Get a list of neighbor indices, edge value names ingoing edge values, or outgoing edge values
        The input format is: [(neighbor index, edge_value_var, value_in, value_out)]
        If a value is not set, a 0 is returned for that neighbor
        """
        if var_type == 'indices':
            return [neighbor[0] if (isinstance(neighbor, tuple) or isinstance(neighbor, list)) \
                    else neighbor \
                    for neighbor in neighbors]
        elif var_type == 'names':
            return [self.edge_values_map[neighbor[1]] for neighbor in neighbors]
        elif var_type == 'ingoing_values':
            return [neighbor[2] if (isinstance(neighbor, tuple) or isinstance(neighbor, list)) \
                    else 0 \
                    for neighbor in neighbors]
        elif var_type == 'outgoing_values':
            return [neighbor[3] if ((isinstance(neighbor, tuple) or isinstance(neighbor, list)) and len(neighbor) == 4) \
                    else 0 \
                    for neighbor in neighbors]

    def inactive_scheme(self, scheme):
        if scheme.lower_bound and scheme.lower_bound > self.current_iteration:
            return True
        elif scheme.upper_bound and scheme.upper_bound <= self.current_iteration:
            return True
        return False

    def calculate_properties(self):
        for prop in self.property_functions:
            if self.current_iteration % prop.iteration_interval == 0:
                property_outputs = self.properties.get(prop.name, [])
                property_outputs.append(prop.execute())
                self.properties[prop.name] = property_outputs

    def get_properties(self):
        return self.properties

    def valid_update_condition_nodes(self, update, scheme_nodes):
        if not update.condition:
            return scheme_nodes
        return update.condition.get_valid_nodes((scheme_nodes, self.node_states, self.adjacency, self.edge_utility))

    def iteration_assignment(self):
        self.node_states = self.new_node_states[:].copy()
        self.edge_utility = self.new_edge_utility[:].copy()
        self.edge_values = self.new_edge_values[:].copy()

        if self.graph_changed:
            self.adjacency = self.new_adjacency[:].copy()
            self.new_graph = nx.convert_matrix.from_numpy_array(self.new_adjacency)
            self.graph = self.new_graph.copy()

    def prepare_next_iteration(self):
        self.current_iteration += 1
        self.graph_changed = False

    def configure_visualization(self, options, output):
        options['state_names'] = self.state_names
        configuration = VisualizationConfiguration(options)
        model_input = (self.graph, self.state_map, output, self.edge_values_map)
        self.visualizer = Visualizer(configuration, model_input)

    def visualize(self, vis_type):
        self.visualizer.visualize(vis_type)

    def clear(self):
        self.state_map = {}
        self.state_names = []

        self.node_states = np.array([])
        self.new_node_states = np.array([])

        self.property_functions = []
        self.properties = {}

        self.schemes: List[Scheme] = [Scheme(lambda graph: graph.nodes, {'args': {'graph': self.graph}, 'lower_bound': 0})]

        self.edge_utility = np.array([])
        self.new_edge_utility = np.array([])

        self.edge_values_map = {}
        self.edge_values_names = []
        self.edge_values = np.array([])
        self.new_edge_values = np.array([])

        self.current_iteration = 0

    def reset(self):
        # Add more model variables here
        self.node_states = np.zeros((len(self.graph.nodes()), len(self.state_names)))
        self.new_node_states = self.node_states[:].copy()

        self.edge_values = np.zeros((len(self.graph.nodes()), len(self.graph.nodes()), len(self.edge_values_names)))
        self.new_edge_values = self.edge_values[:].copy()

        self.current_iteration = 0
