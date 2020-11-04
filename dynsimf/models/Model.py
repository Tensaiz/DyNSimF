from abc import ABCMeta
import tqdm
import copy
import numpy as np
import networkx as nx

from dynsimf.models.Memory import MemoryConfiguration
from dynsimf.models.Memory import MemoryConfigurationType
from dynsimf.models.Update import Update
from dynsimf.models.Update import UpdateType
from dynsimf.models.Update import UpdateConfiguration
from dynsimf.models.Scheme import Scheme
from dynsimf.models.Visualizer import VisualizationConfiguration
from dynsimf.models.Visualizer import Visualizer

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

class Model(object, metaclass=ABCMeta):
    """
    Partial Abstract Class defining a model
    """
    def __init__(self, graph, config=None, seed=None):
        self.graph = graph
        self.new_graph = copy.deepcopy(graph)
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
        self.new_adjacency = self.adjacency[:]
        if self.config.utility:
            self.initialize_utility()

    def add_property_function(self, fun):
        self.property_functions.append(fun)

    def set_states(self, states):
        self.node_states = np.zeros((len(self.graph.nodes()), len(states)))
        self.new_node_states = self.node_states[:]
        self.state_names = states
        for i, state in enumerate(states):
            self.state_map[state] = i

    def set_initial_state(self, initial_state, args=None):
        arguments = args if args else {}
        for state in initial_state.keys():
            val = initial_state[state]
            if hasattr(val, '__call__'):
                self.node_states[:, self.state_map[state]] = val(**arguments)
            else:
                self.node_states[:, self.state_map[state]] = val
        self.new_node_states = self.node_states[:]

    def initialize_utility(self):
        n_nodes = len(self.graph.nodes())
        self.edge_utility = np.zeros((n_nodes, n_nodes))
        self.new_edge_utility = self.edge_utility[:]

    def get_utility(self):
        return self.edge_utility

    def get_nodes_utility(self, nodes):
        return self.edge_utility[nodes]

    def set_initial_utility(self, init_function, params=None):
        params = params if params else {}
        self.edge_utility = init_function(*params)
        self.new_edge_utility = self.edge_utility[:]

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

    def get_all_neighbors(self):
        neighbors = []
        for node in list(self.graph.nodes()):
            neighbors.append(self.get_neighbors(node))
        return neighbors

    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))

    def simulate(self, n, show_tqdm=True):
        self.simulation_output = {
            'states': {},
            'adjacency': {},
            'utility': {}
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
            f.write('# Iteration {0} - ({2}, {3})\n'.format(self.current_iteration, n_nodes, len(self.state_names)))
            np.savetxt(f, self.node_states)

    def write_utility_iteration(self, n_nodes):
        with open(self.config.utility_memory_config.path, 'a') as f:
            f.write('# Iteration {0} - ({2}, {3})\n'.format(self.current_iteration, n_nodes, n_nodes))
            np.savetxt(f, self.edge_utility)

    def write_adjacency_iteration(self, n_nodes):
        with open(self.config.adjacency_memory_config.path, 'a') as f:
            f.write('# Iteration {0} - ({2}, {3})\n'.format(self.current_iteration, n_nodes, n_nodes))
            np.savetxt(f, self.adjacency)

    def store_simulation_step(self):
        self.store_states_iteration()
        self.store_utility_iteration()
        self.store_adjacency_iteration()

    def store_states_iteration(self):
        if self.config.state_memory_config.memory_size != -1 and self.current_iteration % self.config.state_memory_config.memory_interval == 0:
            self.simulation_output['states'][self.current_iteration] = copy.deepcopy(self.node_states)
        if self.config.state_memory_config.memory_size > 0 and len(self.simulation_output) > self.config.state_memory_config.memory_size:
            self.simulation_output['states'] = {}

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
            self.update_utility(update_nodes, update_nodes)
        elif update.update_type == UpdateType.NETWORK:
            self.update_network(update_nodes, updatables)

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
        self.delete_rows_columns('new_edge_utility', removable_nodes)
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
        self.new_edge_utility = np.vstack([self.new_edge_utility, np.zeros(len(self.new_adjacency))])
        self.new_edge_utility = np.append(self.new_edge_utility, np.zeros((len(self.new_adjacency), 1)), axis=1)
        self.new_node_states = np.vstack([self.new_node_states, np.zeros(len(self.state_names))])
        self.graph_changed = True

    def handle_node_initialization(self, index, node):
        """
        :param index int: The index of the node in the graph
        :param node dict: a node dictionary of the form:
            key (str): 'neighbors', value (list[tuple]): (neighbor_index, utility_in, utility_out)
            key (str): 'states', value (dict): { 'state_name' (str) : state_value (number) }
        Initialize a node by setting the utility of the neighbors and setting the node states
        """
        self.set_node_neighbor_utilities(index, node['neighbors'])
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
        neighbors variable format: (neighbor_index, origin_to_neighbor_utility, neighbor_to_origin_utility)
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
        # Clear all ingoing and outgoing utility
        self.new_edge_utility[origin] = 0
        self.new_edge_utility[:, origin] = 0
        # Set given utility
        self.set_node_neighbor_utilities(origin, neighbors)
        self.graph_changed = True

    def handle_adjacency_node_add(self, origin, neighbors):
        neighbor_indices = self.neighbor_update_to_var('indices', neighbors)
        self.new_adjacency[origin, neighbor_indices] = 1
        self.new_adjacency[neighbor_indices, origin] = 1
        self.set_node_neighbor_utilities(origin, neighbors)
        self.graph_changed = True

    def handle_adjacency_node_remove(self, origin, neighbors):
        self.new_adjacency[origin, neighbors] = 0
        self.new_adjacency[neighbors, origin] = 0
        self.new_edge_utility[origin, neighbors] = 0
        self.new_edge_utility[neighbors, origin] = 0
        self.graph_changed = True

    def set_node_neighbor_utilities(self, origin, neighbors):
        neighbor_indices = self.neighbor_update_to_var('indices', neighbors)
        neighbor_outgoing_utility = self.neighbor_update_to_var('ingoing_utility', neighbors)
        neighbor_ingoing_utility = self.neighbor_update_to_var('outgoing_utility', neighbors)
        self.new_edge_utility[origin, neighbor_indices] = neighbor_outgoing_utility
        self.new_edge_utility[neighbor_indices, origin] = neighbor_ingoing_utility

    def neighbor_update_to_var(self, var_type, neighbors):
        """
        Get a list of neighbor indices, ingoing utililty values, or outgoing utility values
        The input format is: [(neighbor index, utility_in, utility_out)]
        If a value is not set, a 0 is returned for that neighbor
        """
        if var_type == 'indices':
            return [neighbor[0] if (isinstance(neighbor, tuple) or isinstance(neighbor, list)) \
                    else neighbor \
                    for neighbor in neighbors]

        elif var_type == 'ingoing_utility':
            return [neighbor[1] if (isinstance(neighbor, tuple) or isinstance(neighbor, list)) \
                    else 0 \
                    for neighbor in neighbors]

        elif var_type == 'outgoing_utility':
            return [neighbor[2] if ((isinstance(neighbor, tuple) or isinstance(neighbor, list)) and len(neighbor) == 3) \
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
        self.node_states = self.new_node_states[:]
        self.edge_utility = self.new_edge_utility[:]

        if self.graph_changed:
            self.adjacency = self.new_adjacency[:]
            self.new_graph = nx.convert_matrix.from_numpy_array(self.new_adjacency)
            self.graph = self.new_graph.copy()

    def prepare_next_iteration(self):
        self.current_iteration += 1
        self.graph_changed = False

    def configure_visualization(self, options, output):
        configuration = VisualizationConfiguration(options)
        self.visualizer = Visualizer(configuration, self.graph, self.state_map, output)

    def visualize(self, vis_type):
        self.visualizer.visualize(vis_type)

    def clear(self):
        self.state_map = {}
        self.state_names = []

        self.node_states = np.array([])
        self.new_node_states = np.array([])

        self.property_functions = []
        self.properties = {}

        self.schemes: List[Scheme] = [Scheme(lambda graph: graph.nodes, {'graph': self.graph}, lower_bound=0)]

        self.edge_utility = np.array([])
        self.new_edge_utility = np.array([])

        self.current_iteration = 0

    def reset(self):
        # Add more model variables here
        self.node_states = np.zeros((len(self.graph.nodes()), len(self.state_names)))
        self.new_node_states = self.node_states[:]
        self.current_iteration = 0
