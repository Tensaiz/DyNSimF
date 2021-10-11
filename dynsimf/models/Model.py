from abc import ABCMeta
import tqdm
import copy
import numpy as np
import networkx as nx
import random

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


class ModelConfiguration(object):
    '''
    Configuration for the model

    Can be used to configure the amount of memory to be kept in memory for the states, adjacency, utility, and edge values.
    The configurations are done using the MemoryConfiguration class.
    '''
    def __init__(self, cfg=None):
        cfg = cfg if cfg else {}
        cfg_keys = list(cfg.keys())

        self.state_memory_config = \
            MemoryConfiguration(MemoryConfigurationType.STATE, {
                'memory_size': 0
            }) \
            if 'state_memory_config' not in cfg_keys \
            else cfg['state_memory_config']
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
            if 'edge_values_memory_config' not in cfg_keys \
            else cfg['edge_values_memory_config']
        self.utility_memory_config = MemoryConfiguration(MemoryConfigurationType.UTILITY) \
            if 'utility_memory_config' not in cfg_keys \
            else cfg['utility_memory_config']

class Model(object, metaclass=ABCMeta):
    '''
    The base class used to define a model
    '''
    def __init__(self, graph, config=None, seed=None):
        '''
        Initialization for the model, set the graph, config, and seed.
        The function automatically initializes all internal variables used for the simulation

        :param graph: The Networkx graph used for the simulation
        :type graph: networkx.graph
        :param config: the configuration object for the model, defaults to ModelConfiguration
        :type config: ModelConfiguration, optional
        :param seed: A seed for the simulation, defaults to None
        :type seed: int, optional
        '''
        self.graph = graph
        self.config = config if config else ModelConfiguration()
        self.clear()
        self.init()
        np.random.seed(seed)
        random.seed(seed)

    @property
    def constants(self):
        '''
        Constants of the model

        :return: Model constants
        :rtype: dict
        '''
        return self.__constants

    @constants.setter
    def constants(self, constants):
        self.__constants = constants

    @property
    def nodes(self):
        return list(self.graph.nodes())

    def init(self):
        '''
        Initialize the internal variables used for the simulation.
        '''
        self.graph_changed = False
        self.adjacency = nx.convert_matrix.to_numpy_array(self.graph)
        self.new_adjacency = self.adjacency.copy()
        self.graph = nx.convert_matrix.from_numpy_array(self.adjacency)
        self.new_graph = copy.deepcopy(self.graph)

    def add_property_function(self, fun):
        '''
        Add a PropertyFunction object to the model

        :param fun: The property function to add to the model
        :type fun: PropertyFunction
        '''
        self.property_functions.append(fun)

    def set_states(self, state_names):
        '''
        Initialize the states of the model,
        an internal representation is made to keep track of the state values
        and the corresponding names.

        :param state_names: A list of strings that contain the names of each state
        :type state_names: list[str]
        '''
        self.node_states = np.zeros((len(self.graph.nodes()), len(state_names)))
        self.new_node_states = self.node_states.copy()
        self.state_names = state_names
        for i, state in enumerate(state_names):
            self.state_map[state] = i

    def set_edge_values(self, edge_values_names):
        '''
        Initialize the edge values of the model,
        an internal representation is made to keep track of the edge values
        and the corresponding names.

        :param edge_values_names: A list of strings that contain the names of each edge value
        :type edge_values_names: list[str]
        '''
        self.edge_values = np.zeros((len(edge_values_names), len(self.graph.nodes()), len(self.graph.nodes())))
        self.new_edge_values = self.edge_values.copy()
        self.edge_values_names = edge_values_names
        for i, edge_values_name in enumerate(edge_values_names):
            self.edge_values_map[edge_values_name] = i

    def set_initial_state(self, initial_state, args=None):
        '''
        Set the initial values for each state for each node.

        :param initial_state: A dictionary containing either an array, function or constant value.\
        The keys should represent state names and their value will be used as the initial value.\
        In case of an array, the indices will be used for the correpsonding nodes.\
        A constant value will be duplicated for each node for that state.\
        If a function is used, it should return either a constant value or array.
        :type initial_state: list or number or function
        :param args: Arguments to provide to every function provided in the initial_state dictionary, defaults to {}
        :type args: dict, optional
        '''
        arguments = args if args else {}
        for state in initial_state.keys():
            val = initial_state[state]
            if hasattr(val, '__call__'):
                self.node_states[:, self.state_map[state]] = val(**arguments)
            else:
                self.node_states[:, self.state_map[state]] = val
        self.new_node_states = self.node_states.copy()

    def set_initial_edge_values(self, initial_edge_values, args=None):
        '''
        Set the initial values for each edge value for each node connection.

        :param initial_state: A dictionary containing either an array, function or constant value.\
        The keys should represent edge value names and their value will be used as the initial value.\
        In case of an array, it should be 2d and each index should match the value for the edge value between node i to j.\
        A constant value will be duplicated for each node value for that edge value name.\
        If a function is used, it should return either a constant value or a 2d array.
        :type initial_state: list or number or function
        :param args: Arguments to provide to every function provided in the initial_state dictionary, defaults to {}
        :type args: dict, optional
        '''
        arguments = args if args else {}
        for edge_value_name in initial_edge_values.keys():
            val = initial_edge_values[edge_value_name]
            if hasattr(val, '__call__'):
                self.edge_values[self.edge_values_map[edge_value_name], :, :] = val(**arguments)
            else:
                self.edge_values[self.edge_values_map[edge_value_name], :, :] = val
        self.new_edge_values = self.edge_values.copy()

    def get_state_index(self, state):
        '''
        Get the internal index for a given state name

        :param state: Name of the state to retrieve index from
        :type state: str
        :return: The internal index of the state
        :rtype: int
        '''
        return self.state_map[state]

    def get_state(self, state):
        '''
        Get all values for a given state

        :param state: The state to get the values for each node
        :type state: str
        :return: An array containing a value per node for the given state
        :rtype: numpy.ndarray
        '''
        return self.node_states[:, self.state_map[state]]

    def get_new_state(self, state):
        '''
        Get all values for a given state for the current iteration

        :param state: The state to get the values for each node
        :type state: str
        :return: An array containing a value per node for the given state
        :rtype: numpy.ndarray
        '''
        return self.new_node_states[:, self.state_map[state]]

    def get_node_states(self, node):
        '''
        Get all states of a single node

        :param node: The node to get all states for
        :type node: int
        :return: An array containing values of all states for one node.
        :rtype: numpy.ndarray
        '''
        return self.node_states[node]

    def get_node_state(self, node, state):
        '''
        Get the value for a single state for a single node

        :param node: The node to get the state for
        :type node: int
        :param state: The state name to get the value of
        :type state: str
        :return: The value for the given node's state
        :rtype: number
        '''
        return self.node_states[node, self.state_map[state]]

    def get_node_new_state(self, node, state):
        '''
        Get the value for a single state for a single node for the current iteration

        :param node: The node to get the state for
        :type node: int
        :param state: The state name to get the value of
        :type state: str
        :return: The value for the given node's state
        :rtype: number
        '''
        return self.new_node_states[node, self.state_map[state]]

    def get_nodes_state(self, nodes, state):
        '''
        Get the value for a single state for a list of nodes

        :param node: A list of nodes to get the state for
        :type node: list[int]
        :param state: The state name to get the value of
        :type state: str
        :return: The values for the given node's state
        :rtype: list[number]
        '''
        return self.node_states[nodes, self.state_map[state]]

    def get_nodes_states(self):
        '''
        Get all the states of all the nodes in a 2d array.\n
        Each row represents a node and each column represents a state.

        :return: The 2d array with all the nodes' states
        :rtype: numpy.ndarray
        '''
        return self.node_states

    def get_previous_nodes_states(self, n):
        '''
        Get all the nodes' states from the n'th previous saved iteration

        :param n: The n'th previous iteration to get the states from
        :type n: int
        :return: 2d array with nodes' states from the current iteration minus n
        :rtype: numpy.ndarray
        '''
        available_iterations = list(self.simulation_output['states'].keys())
        return self.simulation_output['states'][available_iterations[-n - 1]]

    def get_previous_nodes_adjacency(self, n):
        '''
        Get all the adjacency matrix from the n'th previous saved iteration
        '''
        available_iterations = list(self.simulation_output['adjacency'].keys())
        return self.simulation_output['adjacency'][available_iterations[-n - 1]]

    def get_edge_values(self, edge_values_name):
        '''
        Get the given edge values for each node

        :param edge_values_name: The name of the edge values to get
        :type edge_values_name: str
        :return: a 2d array with the edge values between nodes. It resembles an adjacency matrix.
        :rtype: numpy.ndarray
        '''
        return self.edge_values[self.edge_values_map[edge_values_name]]

    def get_all_edge_values(self):
        '''
        Get all edge values for each node.\n
        Each index in the returned array contains a 2d matrix with edge values that resembles an adjacency matrix.

        :return: The 3d array with matrices for each edge value name
        :rtype: numpy.ndarray
        '''
        return self.edge_values

    def add_update(self, fun, args=None, condition=None, get_nodes=False, update_type=None):
        '''
        Add an update function to the model to update the states, utility, edge values, or network of the model.

        :param fun: The function to be executed and get the new state values from.\n
        The function should return a dictionary where each key matches a state added using the set_states function.\n
        The value for each key should either be a constant value,
        or an array, where each value should be the value of a node in the network.\n
        The indices in the array match the indices of the nodes in the networkx graph.
        :type fun: function
        :param args: Arguments for the given function. The keys should match the parameters of the function, defaults to None
        :type args: dict, optional
        :param condition: A single chained condition that filters the nodes.
        The update will only be applied on the nodes that the condition returns, defaults to None
        :type condition: Condition, optional
        :param get_nodes: A boolean indicating whether the given function
        should receive a list argument that gives the indices of the nodes
        that the update will be applied on, defaults to False
        :type get_nodes: bool, optional
        :param update_type: The type of update, could be a state, network, edge values, or utility update. defaults to None
        :type update_type: UpdateType, optional
        '''
        arguments = args if args else {}
        if condition:
            self.set_conditions_state_indices(condition)
        update_type = update_type if update_type else UpdateType.STATE
        update = self.create_update((fun, arguments, condition, get_nodes, update_type))
        self.schemes[0].add_update(update)

    def set_conditions_state_indices(self, condition):
        '''
        An internal function that is used to set the indices for each state in every condition when the simulation has started.
        This is used because initially only the state names are provided when the conditions are created.
        The function cycles through each condition and checks whether a state index should be set.

        :param condition: The first condition in the condition chain to set the state indices of
        :type condition: Condition
        '''
        current_condition = condition
        while current_condition:
            if current_condition.get_state():
                current_condition.set_state_index(self.state_map[current_condition.get_state()])
            current_condition = current_condition.chained_condition

    def add_state_update(self, fun, args=None, condition=None, get_nodes=False):
        '''
        Helper function that is a shorthand to set an update to update states.
        It shows more clearly what is updated in the name and does not require an UpdateType object.

        :param fun: The function to be executed and get the new state values from.\n
        The function should return a dictionary where each key matches a state added using the set_states function.\n
        The value for each key should either be a constant value,
        or an array, where each value should be the value of a node in the network.\n
        The indices in the array match the indices of the nodes in the networkx graph.
        :type fun: function
        :param args: Arguments for the given function. The keys should match the parameters of the function, defaults to None
        :type args: dict, optional
        :param condition: A single chained condition that filters the nodes.
        The update will only be applied on the nodes that the condition returns, defaults to None
        :type condition: Condition, optional
        :param get_nodes: A boolean indicating whether the given function
        should receive a list argument that gives the indices of the nodes
        that the update will be applied on, defaults to False
        :type get_nodes: bool, optional
        '''
        self.add_update(fun, args, condition, get_nodes, UpdateType.STATE)

    def add_network_update(self, fun, args=None, condition=None, get_nodes=False):
        '''
        Helper function that is a shorthand to set an update to update the network.\
        It shows more clearly what is updated in the name and does not require an UpdateType object.

        :param fun: The function that updates the network.\
        To update the network using a function, \
        a dictionary should be returned with specific keys indicating what kind of change the values of the dictionary are. \
        There are three options: `add`, `edge_change`, and `remove`. The remove option is the most straightforward, \
        if this key is included in the dictionary that is returned by the function, \
        then the corresponding value should be a list of the names of all the nodes that should be removed from the network. \
        When using the `add` key, the corresponding value should be a list of dictionaries, \
        where each dictionary inside the list will serve as the initialization for a new node. \
        This dictionary should contain 2 keys: `neighbors` and `states`. \
        The `neighbors` key should have a list as value, where each entry in the list should be a name of a neighbor. \
        This way new nodes can immediately be connected to other nodes when they are added to the network. \
        The `states` key should have another dictionary as a value, \
        in which every key can refer to a state with corresponding initial values for that node. \
        This way, nodes can now be added to the network and their connections and states can immediately be initialized.\n

        :type fun: function
        :param args: Arguments for the given function. The keys should match the parameters of the function, defaults to None
        :type args: dict, optional
        :param condition: A single chained condition that filters the nodes.\
        The update will only be applied on the nodes that the condition returns, defaults to None
        :type condition: Condition, optional
        :param get_nodes: A boolean indicating whether the given function\
        should receive a list argument that gives the indices of the nodes\
        that the update will be applied on, defaults to False
        :type get_nodes: bool, optional
        '''
        self.add_update(fun, args, condition, get_nodes, UpdateType.NETWORK)

    def add_edge_values_update(self, fun, args=None, condition=None, get_nodes=False):
        '''
        Helper function that is a shorthand to set an update to update the edge values.\
        It shows more clearly what is updated in the name and does not require an UpdateType object.

        :param fun: The function that updates the edge values. It should return a 2d array with values for each desired edge value
        :type fun: function
        :param args: Arguments for the given function. The keys should match the parameters of the function, defaults to None
        :type args: dict, optional
        :param condition: A single chained condition that filters the nodes.\
        The update will only be applied on the nodes that the condition returns, defaults to None
        :type condition: Condition, optional
        :param get_nodes: A boolean indicating whether the given function\
        should receive a list argument that gives the indices of the nodes\
        that the update will be applied on, defaults to False
        :type get_nodes: bool, optional
        '''
        self.add_update(fun, args, condition, get_nodes, UpdateType.EDGE_VALUES)

    def create_update(self, update_content):
        '''
        Internal function used to create an Update object

        :param update_content: The parameters required for an update object
        :type update_content: tuple or list
        :return: The update object created from the update content
        :rtype: Update
        '''
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
        '''
        Adds a scheme to the model

        :param scheme: The Scheme object to add to the model
        :type scheme: Scheme
        '''
        self.set_scheme_update_condition_state_indices(scheme)
        self.schemes.append(scheme)

    def set_scheme_update_condition_state_indices(self, scheme):
        '''
        Internal function that sets the state indices for each condition that has them in any scheme

        :param scheme: Scheme to set the conditions' state indices for
        :type scheme: Scheme
        '''
        if scheme.updates:
            for update in scheme.updates:
                if update.condition:
                    self.set_conditions_state_indices(update.condition)

    def get_adjacency(self):
        '''
        Get the adjacency matrix of the networkx graph

        :return: The adjacency matrix, which is a 2d numpy array
        :rtype: numpy.ndarray
        '''
        return self.adjacency

    def get_nodes_adjacency(self, nodes):
        '''
        Get the adjacency values for a specific subset of nodes

        :param nodes: A list of nodes to get the adjacency for
        :type nodes: list or numpy.ndarray
        :return: A numpy.ndarray where each row matches a node from the argument and each column represents whether the node is linked to it.\
        A 1 indicates a connection and a 0 the opposite.
        :rtype: numpy.ndarray
        '''
        return self.adjacency[nodes]

    def get_neighbors(self, node):
        '''
        Get a list of neighbors of one node

        :param node: The node to get the neighbors of
        :type node: int
        :return: A list of node names (indices) that are adjacent to the given node
        :rtype: list
        '''
        return list(self.graph.neighbors(node))

    def get_neighbors_neighbors_adjacency_matrix(self):
        '''
        Returns a matrix that indicates which nodes are only connected through another node.

        :return: A 2d numpy array where each 1 represents that the node in that row is a neighbor of a neighbor of the node in that column.
        :rtype: numpy.ndarray
        '''
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
        '''
        Get an array that shows the which neighbors the neighbors of a given node have.

        :param node: The node to get the neighbors' neighbors of
        :type node: int
        :return: An array where each one represents that the node in that column is a neighbor of a neighbor of the given node
        :rtype: numpy.ndarray
        '''
        neighbors_neighbors_matrix = self.get_neighbors_neighbors_adjacency_matrix()
        return np.array(self.graph.nodes)[np.where(neighbors_neighbors_matrix[node] > 0)[0]]

    def simulate(self, n, show_tqdm=True):
        '''
        Simulate the model `n` iterations.

        :param n: The amount of iterations to run the simulation
        :type n: int
        :param show_tqdm: Shows a progress bar and estimates time until completion for the simulation, defaults to True
        :type show_tqdm: bool, optional
        :return: Simulation output. A dictionary is returned with 3 keys: `states`, `adjacency`, and `edge_values`.\
        Each value linked to a key is another dictionary with keys for each iteration of the simulation.\
        The value for each iteration key contains the relevant values for that key for that iteration.
        :rtype: dict
        '''
        self.simulation_output = {
            'states': {},
            'adjacency': {},
            'edge_values': {}
        }
        self.prepare_output(n)
        self.simulation_steps(n, show_tqdm)
        return self.simulation_output

    def prepare_output(self, n):
        '''
        Internal function used to create files that are used to store intermediate results of the simulation.

        :param n: Total amount of simulation iterations
        :type n: int
        '''
        n_nodes = len(self.graph.nodes)
        if self.config.state_memory_config.save_disk:
            with open(self.config.state_memory_config.path, 'w') as f:
                f.write(self.config.state_memory_config.get_description_string(n, n_nodes, len(self.state_names)))
        if self.config.adjacency_memory_config.save_disk:
            with open(self.config.adjacency_memory_config.path, 'w') as f:
                f.write(self.config.adjacency_memory_config.get_description_string(n, n_nodes, n_nodes))

    def simulation_steps(self, n, show_tqdm):
        '''
        Store the initial state of the simulation and then execute the iteration_step function each iteration.

        :param n: Total amount of iterations
        :type n: int
        :param show_tqdm: Shows a progress bar and estimates time until completion for the simulation
        :type show_tqdm: bool
        '''
        self.store_simulation_step()
        if show_tqdm:
            for _ in tqdm.tqdm(range(0, n)):
                self.simulation_step()
        else:
            for _ in range(0, n):
                self.simulation_step()

    def simulation_step(self):
        '''
        A single iteration step.
        The iteration is completed and then stored or written if so configured.
        '''
        self.iteration()
        self.write_simulation_step()
        self.store_simulation_step()

    def write_simulation_step(self):
        '''
        Write the output of the simulation to the configured files if the iteration matches the configured save interval.
        '''
        n_nodes = len(self.graph.nodes)
        if self.config.state_memory_config.save_disk and self.current_iteration % self.config.state_memory_config.save_interval == 0:
            self.write_states_iteration(n_nodes)
        if self.config.adjacency_memory_config.save_disk and self.current_iteration % self.config.adjacency_memory_config.save_interval == 0:
            self.write_adjacency_iteration(n_nodes)

    def write_states_iteration(self, n_nodes):
        '''
        Write the states of the current iteration to a configured file

        :param n_nodes: The amount of nodes in the simulation
        :type n_nodes: int
        '''
        with open(self.config.state_memory_config.path, 'a') as f:
            f.write('# Iteration {0} - ({1}, {2})\n'.format(self.current_iteration, n_nodes, len(self.state_names)))
            np.savetxt(f, self.node_states)

    def write_edge_values_iteration(self, n_nodes):
        pass

    def write_adjacency_iteration(self, n_nodes):
        '''
        Write the adjacency matrix of the current iteration to a configured file

        :param n_nodes: The amount of nodes in the simulation
        :type n_nodes: int
        '''
        with open(self.config.adjacency_memory_config.path, 'a') as f:
            f.write('# Iteration {0} - ({1}, {2})\n'.format(self.current_iteration, n_nodes, n_nodes))
            np.savetxt(f, self.adjacency)

    def store_simulation_step(self):
        '''
        Store the states, adjacency, and edge values for the current iteration
        '''
        self.store_states_iteration()
        self.store_adjacency_iteration()
        self.store_edge_values_iteration()

    def store_states_iteration(self):
        '''
        Store the states for the current iteration in memory. If the configured memory size is exceeded, clear the memory.
        '''
        if self.config.state_memory_config.memory_size != -1 and self.current_iteration % self.config.state_memory_config.memory_interval == 0:
            self.simulation_output['states'][self.current_iteration] = copy.deepcopy(self.node_states)
        if self.config.state_memory_config.memory_size > 0 and len(self.simulation_output) > self.config.state_memory_config.memory_size:
            self.simulation_output['states'] = {}

    def store_edge_values_iteration(self):
        '''
        Store the edge values for the current iteration in memory. If the configured memory size is exceeded, clear the memory.
        '''
        if self.config.edge_values_memory_config.memory_size != -1 and self.current_iteration % self.config.edge_values_memory_config.memory_interval == 0:
            self.simulation_output['edge_values'][self.current_iteration] = copy.deepcopy(self.edge_values)
        if self.config.edge_values_memory_config.memory_size > 0 and len(self.simulation_output) > self.config.edge_values_memory_config.memory_size:
            self.simulation_output['edge_values'] = {}

    def store_adjacency_iteration(self):
        '''
        Store the adjacency matrix of the current iteration in memory. If the configured memory size is exceeded, clear the memory.
        '''
        if self.config.adjacency_memory_config.memory_size != -1 and self.current_iteration % self.config.adjacency_memory_config.memory_interval == 0:
            self.simulation_output['adjacency'][self.current_iteration] = copy.deepcopy(self.adjacency)
        if self.config.adjacency_memory_config.memory_size > 0 and len(self.simulation_output) > self.config.adjacency_memory_config.memory_size:
            self.simulation_output['adjacency'] = {}

    def iteration(self):
        '''
        A single iteration step.
        All values are calculated and assigned to their variables.
        The property functions of the model are executed and the next iteration is prepared.
        '''
        self.iteration_calculation()
        self.iteration_assignment()
        self.calculate_properties()
        self.prepare_next_iteration()

    def iteration_calculation(self):
        '''
        The calculation part of an iteration.
        Every active scheme is evaluated; every update in the scheme is executed and the results are assigned.
        '''
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
        '''
        Function that runs the correct update execution function based on the type of the update

        :param update: The update function
        :type update: Update
        :param update_nodes: The nodes that the update function is applied on
        :type update_nodes: list or numpy.ndarray
        :param updatables: Dictionary with keys as states or network keywords and values which should be assigned to the nodes
        :type updatables: dict
        '''
        if update.update_type == UpdateType.STATE:
            self.update_state(update_nodes, updatables)
        elif update.update_type == UpdateType.NETWORK:
            self.update_network(update_nodes, updatables)
        elif update.update_type == UpdateType.EDGE_VALUES:
            self.update_edge_values(update_nodes, updatables)

    def update_state(self, nodes, updatables):
        '''
        Update all states from an update for the given nodes

        :param nodes: nodes that the update should be assigned to
        :type nodes: list or numpy.ndarray
        :param updatables: Dictionary with keys as states and values which should be assigned to the nodes for that state
        :type updatables: dict
        '''
        for state, update_output in updatables.items():
            if isinstance(update_output, list) or \
               isinstance(update_output, np.ndarray) or \
               isinstance(update_output, int) or \
               isinstance(update_output, float):
                self.new_node_states[nodes, self.state_map[state]] = update_output
            elif isinstance(update_output, dict):
                # Add a 2d array implementation instead of for loop
                for node, values in update_output.items():
                    self.new_node_states[node, self.state_map[state]] = values

    def update_edge_values(self, update_nodes, updatables):
        '''
        Update the edge values for each edge value returned by an update

        :param update_nodes: the nodes to update
        :type update_nodes: numpy.ndarray or list
        :param updatables: Dictionary with keys as edge value names and 2d array values which should be assigned to the nodes
        :type updatables: dict
        '''
        for edge_values_name, update_output in updatables.items():
            self.new_edge_values[self.edge_values_map[edge_values_name], update_nodes] = update_output

    def update_network(self, update_nodes, updatables):
        '''
        Assign the correct network update

        :param update_nodes: the nodes to update
        :type update_nodes: numpy.ndarray or list
        :param updatables: Dictionary with keys as network update keywords and relevant values for that network update
        :type updatables: dict
        '''
        for network_update_type, change in updatables.items():
            self.assign_network_operation(network_update_type, change, update_nodes)

    def assign_network_operation(self, network_update_type, change, update_nodes):
        '''
        Assign a network operation depending on whether the network update removes or adds nodes, or changes the adjacencies of the nodes.

        :param network_update_type: The type of the network update
        :type network_update_type: str
        :param change: The first parameter of the specified network update
        :type change: any
        :param update_nodes: The update to perform on the nodes
        :type update_nodes: any
        '''
        network_update_type_to_function = {
            'remove': self.network_nodes_remove,
            'add': self.network_nodes_add,
            'edge_change': self.network_edges_change
        }
        network_update_type_to_function[network_update_type](change, update_nodes)

    def network_nodes_remove(self, removable_nodes, _):
        '''
        Remove a list of nodes from the network

        :param removable_nodes: The nodes to remove from the network
        :type removable_nodes: numpy.ndarray or list
        '''
        self.new_node_states = np.delete(self.new_node_states, removable_nodes, axis=0)
        self.delete_rows_columns('new_adjacency', removable_nodes)

    def delete_rows_columns(self, var, removables):
        '''
        Helper function to delete a row and column from a 2d numpy array, used for removing nodes

        :param var: The variable to delete the column and row of, should be 2d numpy array
        :type var: numpy.ndarray
        :param removables: Indices of the rows and columns to delete
        :type removables: numpy.ndarray or list
        '''
        setattr(self, var, np.delete(getattr(self, var), removables, axis=0))
        setattr(self, var, np.delete(getattr(self, var), removables, axis=1))

    def network_nodes_add(self, new_nodes, _):
        '''
        Add a list of new node dictionaries to the model

        :param new_nodes: A list of dictionaries for new nodes
        :type new_nodes: list[dict]
        '''
        node_index = len(self.new_node_states)
        for node in new_nodes:
            self.initialize_new_node()
            self.handle_node_initialization(node_index, node)
            node_index += 1

    def initialize_new_node(self):
        '''
        Add a new row of 0s to the new adjacency, new edge utilities and new node states matrices
        Also add a new column of 0s to the new adjacency and new edge utilities
        '''
        self.new_adjacency = np.vstack([self.new_adjacency, np.zeros(len(self.new_adjacency))])
        self.new_adjacency = np.append(self.new_adjacency, np.zeros((len(self.new_adjacency), 1)), axis=1)
        self.new_node_states = np.vstack([self.new_node_states, np.zeros(len(self.state_names))])
        self.graph_changed = True

    def handle_node_initialization(self, index, node):
        '''
        Initialize a node by setting the edge values of the neighbors and setting the node states

        :param index int: The index of the node in the graph
        :param node dict: a node dictionary of the form:
            key (str): 'neighbors', value (list[tuple]): (neighbor_index, edge_values_name values_in, values_out)
            key (str): 'states', value (dict): { 'state_name' (str) : state_value (number) }
        '''
        self.set_node_neighbor_values(index, node['neighbors'])
        self.set_new_node_states(index, node['states'])

    def set_new_node_states(self, index, states):
        '''
        Set the values for each state of the new node

        :param index: Index of the new node
        :type index: int
        :param states: An array containing values for each state for a node
        :type states: np.ndarray or list
        '''
        for state, value in states.items():
            self.new_node_states[index, self.state_map[state]] = value

    def network_edges_change(self, change, _):
        '''
        Change the edges of a network by adding and removing links.

        :param change: Dictionary with the changes to apply on the network\n
            key (str): 'overwrite', value (list[tuple]): [(neighbor, edge value name, util_in, util_out)]
            key (str): 'add', value (list[tuple]): [(neighbor, edge value name, util_in, util_out)] or [neighbor, neighbor 2, neighborN]
            key (str): 'remove', value (list): [neighbor index, neighbor 2 index, neighbor 3 index]
        :type change: dict
        '''
        for origin, node_changes in change.items():
            for adjacency_change_type, neighbors in node_changes.items():
                if adjacency_change_type == 'overwrite':
                    self.handle_adjacency_node_overwrite(origin, neighbors)
                elif adjacency_change_type == 'add':
                    self.handle_adjacency_node_add(origin, neighbors)
                elif adjacency_change_type == 'remove':
                    self.handle_adjacency_node_remove(origin, neighbors)

    def handle_adjacency_node_overwrite(self, origin, neighbors):
        '''
        Overwrite the adjacency of nodes

        :param origin: Origin node index to change adjacency for
        :type origin: int
        :param neighbors: Neighbor indices and optional edge values for their link\n
        Format: (neighbor_index, edge_variable_name, origin_to_neighbor_val, neighbor_to_origin_val)
        :type neighbors: list[tuple]
        '''
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
        # Set given edge values
        self.set_node_neighbor_values(origin, neighbors)
        self.graph_changed = True

    def handle_adjacency_node_add(self, origin, neighbors):
        '''
        Set new node adjacency and if applicable assign given edge values

        :param origin: Index of the origin node
        :type origin: int
        :param neighbors: Format: list[(neighbor_index, edge_variable_name, origin_to_neighbor_val, neighbor_to_origin_val)]
        :type neighbors: list[tuple]
        '''
        neighbor_indices = self.neighbor_update_to_var('indices', neighbors)
        self.new_adjacency[origin, neighbor_indices] = 1
        self.new_adjacency[neighbor_indices, origin] = 1
        if len(neighbors) > 0 and isinstance(neighbors[0], tuple):
            self.set_node_neighbor_values(origin, neighbors)
        self.graph_changed = True

    def set_node_neighbor_values(self, origin, neighbors):
        '''
        Set the neighbors for an origin node and their edge values if given

        :param origin: Origin node index
        :type origin: int
        :param neighbors: Array of neighbor indices and their edge values in a tuple if given\n
        Format: [(neighbor index, edge_value_var, value_in, value_out)]
        :type neighbors: list[tuple] or list or numpy.ndarray
        '''
        neighbor_indices = self.neighbor_update_to_var('indices', neighbors)
        edge_value_names = self.neighbor_update_to_var('names', neighbors)
        neighbor_ingoing_values = self.neighbor_update_to_var('ingoing_values', neighbors)
        neighbor_outgoing_values = self.neighbor_update_to_var('outgoing_values', neighbors)
        self.new_edge_values[edge_value_names, origin, neighbor_indices] = neighbor_outgoing_values
        self.new_edge_values[edge_value_names, neighbor_indices, origin] = neighbor_ingoing_values

    def handle_adjacency_node_remove(self, origin, neighbors):
        '''
        Remove neighbor links of an origin node

        :param origin: Origin node index to remove the connections from
        :type origin: int
        :param neighbors: Array of neighbors to remove
        :type neighbors: list or numpy.ndarray
        '''
        self.new_adjacency[origin, neighbors] = 0
        self.new_adjacency[neighbors, origin] = 0
        for edge_value_name in self.edge_values_names:
            self.new_edge_values[self.edge_values_map[edge_value_name], origin, neighbors] = 0
            self.new_edge_values[self.edge_values_map[edge_value_name], neighbors, origin] = 0
        self.graph_changed = True

    def neighbor_update_to_var(self, var_type, neighbors):
        '''
        Get a list of neighbor indices, edge value names ingoing edge values, or outgoing edge values
        The input format is: [(neighbor index, edge_value_var, value_in, value_out)]
        If a value is not set, a 0 is returned for that neighbor
        '''
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
        '''
        Returns True when a scheme is not active this iteration

        :param scheme: Scheme to check whether it's active
        :type scheme: Scheme
        :return: Bool that indicates whether the scheme is active or not
        :rtype: bool
        '''
        if scheme.lower_bound and scheme.lower_bound > self.current_iteration:
            return True
        elif scheme.upper_bound and scheme.upper_bound <= self.current_iteration:
            return True
        return False

    def calculate_properties(self):
        '''
        Calculate the output from the added property functions and append their results
        '''
        for prop in self.property_functions:
            if self.current_iteration % prop.iteration_interval == 0:
                property_outputs = self.properties.get(prop.name, [])
                property_outputs.append(prop.execute())
                self.properties[prop.name] = property_outputs

    def get_properties(self):
        '''
        Get all properties from the PropertyFunctions of the model

        :return: Dictionary with values for the different property functions, \
        keys are the names and values a list of outputs of the functions as values
        :rtype: dict
        '''
        return self.properties

    def valid_update_condition_nodes(self, update, scheme_nodes):
        '''
        [summary]

        :param update: The update to get the eligible nodes for
        :type update: Update
        :param scheme_nodes: All eligible nodes from the scheme
        :type scheme_nodes: list or numpy.ndarray
        :return: List of nodes that pass the condition
        :rtype: list or numpy.ndarray
        '''
        if not update.condition:
            return scheme_nodes
        return update.condition.get_valid_nodes((scheme_nodes, self.node_states, self.adjacency, self.edge_utility))

    def iteration_assignment(self):
        '''
        Assign all values from the iteration output
        '''
        self.node_states = self.new_node_states.copy()
        self.edge_values = self.new_edge_values.copy()

        if self.graph_changed:
            self.adjacency = self.new_adjacency.copy()
            self.new_graph = nx.convert_matrix.from_numpy_array(self.new_adjacency)
            self.graph = self.new_graph.copy()

    def prepare_next_iteration(self):
        '''
        Increase the iteration counter and prepare the next iteration by setting graph change flag to false
        '''
        self.current_iteration += 1
        self.graph_changed = False

    def configure_visualization(self, options, output):
        '''
        Configure the visualization using a dictionary and the output of a simulation

        :param options: Dictionary with the visualization options
        :type options: dict
        :param output: Output of a simulation
        :type output: dict
        '''
        options['state_names'] = self.state_names
        configuration = VisualizationConfiguration(options)
        model_input = (self.graph, self.state_map, output, self.edge_values_map)
        self.visualizer = Visualizer(configuration, model_input)

    def visualize(self, vis_type):
        '''
        Visualize the model results using the Visualization class

        :param vis_type: The visualization to use
        :type vis_type: str
        '''
        self.visualizer.visualize(vis_type)

    def clear(self):
        '''
        Initialize all internal variables as empty
        '''
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
        '''
        Initialize internal variables; state and edge values to 0
        '''
        self.node_states = np.zeros((len(self.graph.nodes()), len(self.state_names)))
        self.new_node_states = self.node_states.copy()

        self.edge_values = np.zeros((len(self.graph.nodes()), len(self.graph.nodes()), len(self.edge_values_names)))
        self.new_edge_values = self.edge_values.copy()

        self.current_iteration = 0
