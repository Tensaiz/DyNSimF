import copy
from enum import Enum

import numpy as np

from dynsimf.models.Model import Model

from typing import List

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class FunctionType(Enum):
    MATRIX = 0
    PAIRWISE = 1


class SampleMethod(Enum):
    ALL = 0
    NEIGHBORS_OF_NEIGHBORS = 1
    CUSTOM = 2


class UtilityCostModel(Model):
    def __init__(self, graph, cost_threshold, config=None, seed=None):
        super().__init__(graph, config, seed)
        self.cost_threshold = cost_threshold
        self.initialize_user_input()
        self.initialize_utility()
        self.initialize_cost()

    def initialize_user_input(self):
        self.utility_function = None
        self.utility_function_type = None

        self.cost_function = None
        self.cost_function_type = None

        self.sample_method = None
        self.custom_sample_function = None

    def initialize_utility(self):
        n_nodes = len(self.graph.nodes())
        self.edge_utility = np.zeros((n_nodes, n_nodes))
        self.new_edge_utility = self.edge_utility.copy()

    def initialize_cost(self):
        n_nodes = len(self.graph.nodes())
        self.cost = np.zeros((n_nodes, n_nodes))
        self.new_cost = self.cost.copy()

    """
    Initial utility cost model configuration
    """

    def add_utility_function(self, function, function_type):
        self.utility_function = function
        self.utility_function_type = function_type

    def add_cost_function(self, function, function_type):
        self.cost_function = function
        self.cost_function_type = function_type

    def set_sampling_function(self, sample_method, function=None):
        self.sample_method = sample_method
        self.custom_sample_function = function
        if self.sample_method == SampleMethod.CUSTOM and function is None:
            raise ValueError('To use the custom sample method, a function argument must be provided')

    """
    Iteration functions
    """

    def simulate(self, n, show_tqdm=True):
        self.check_user_input_set()
        self.simulation_output = {
            'states': {},
            'adjacency': {},
            'utility': {},
            'edge_values': {}
        }
        self.prepare_output(n)
        self.prepare_simulation()
        super().simulation_steps(n, show_tqdm)
        return self.simulation_output

    def check_user_input_set(self):
        if self.utility_function is None or self.utility_function_type is None:
            raise ValueError('The utility function and function type should be set using the `add_utility_function` function, before calling simulate!')

        if self.cost_function is None or self.cost_function_type is None:
            raise ValueError('The cost function and function type should be set using the `add_cost_function` function, before calling simulate!')

        if self.sample_method is None:
            raise ValueError('The sample method must be set using the `set_sampling_function` function, before calling simulate!')

    def prepare_simulation(self):
        self.calculate_utility()
        self.edge_utility = self.new_edge_utility.copy()
        self.calculate_cost()
        self.cost = self.new_cost.copy()

    def iteration(self):
        super().iteration_calculation()
        self.calculate_utility()
        self.calculate_cost()
        self.optimize_nodes_utility()
        self.iteration_assignment()
        super().calculate_properties()
        super().prepare_next_iteration()

    def calculate_utility(self):
        if self.utility_function_type == FunctionType.MATRIX:
            self.new_edge_utility = self.utility_function()
        elif self.utility_function_type == FunctionType.PAIRWISE:
            self.calculate_pairwise(self.new_edge_utility, self.utility_function)

    def calculate_cost(self):
        if self.cost_function_type == FunctionType.MATRIX:
            self.new_cost = self.cost_function()
        elif self.cost_function_type == FunctionType.PAIRWISE:
            self.calculate_pairwise(self.new_cost, self.cost_function)

    def iteration_assignment(self):
        super().iteration_assignment()
        self.edge_utility = self.new_edge_utility.copy()
        self.cost = self.new_cost.copy()

    def calculate_pairwise(self, variable, function):
        nodes = list(self.graph.nodes)
        if self.sample_method == SampleMethod.ALL:
            self.calculate_all_pairwise(nodes, variable, function)
        elif self.sample_method == SampleMethod.NEIGHBORS_OF_NEIGHBORS:
            # Calculate cost of neighbors pairwise
            self.calculate_neighbors_pairwise(nodes, variable, function)
            # Calculate costs of neighbors of neighbors pairwise
            self.calculate_fof_pairwise(nodes, variable, function)
        elif self.sample_method == SampleMethod.CUSTOM:
            # TODO: Implement custom sampling
            print('Custom sampling has not been implemented for pairwise comparisons yet')
            exit()

    def calculate_all_pairwise(self, nodes, variable, function):
        """
        Calculate the function parameter for the variable parameter pairwise for every node
        """
        for i, node in enumerate(nodes):
            for j, other_node in enumerate(nodes):
                if node == other_node:
                    continue
                variable[i, j] = function(node, other_node)

    def calculate_neighbors_pairwise(self, nodes, variable, function):
        for i, node in enumerate(nodes):
            neighbors = super().get_neighbors(node)
            for neighbor in neighbors:
                variable[i, nodes.index(neighbor)] = function(node, neighbor)

    def calculate_fof_pairwise(self, nodes, variable, function):
        """
        Calculate the function parameter for the variable paramater, pairwise for every node and their neighbors of neighbors
        """
        for i, node in enumerate(nodes):
            neighbors_of_neighbors = super().get_neighbors_neighbors(node)
            for nb_of_nb in neighbors_of_neighbors:
                variable[i, nodes.index(nb_of_nb)] = function(node, nb_of_nb)

    def optimize_nodes_utility(self):
        """
        Optimize the utility for each node
        Nodes that exceed the cost_threshold remove a link with lowest utility to achieve < threshold
        Nodes that don't exceed the threshold add a new connection with highest utility while staying < threshold
        TODO: Optimize by reconfiguring existing links by checking neighborhood
        TODO: Do the above to do for nodes for which cost_sum == cost_threshold
        """
        adjacency = super().get_adjacency()
        # Get cost of current connections and sum per node
        total_cost_per_node = (self.new_cost * adjacency).sum(axis=1)
        # Remove a connection from nodes that surpass the cost threshold
        nodes_above_cost_threshold = np.where(total_cost_per_node > self.cost_threshold)[0]
        if len(nodes_above_cost_threshold) > 0:
            self.remove_node_connections(nodes_above_cost_threshold)
        # Let nodes that still have cost room add a connection
        nodes_below_cost_threshold = np.where(total_cost_per_node < self.cost_threshold)[0]
        if len(nodes_below_cost_threshold) > 0:
            self.add_node_connections(nodes_below_cost_threshold)

    def remove_node_connections(self, nodes):
        """
        Remove a neighbor for each node to get the total cost under the cost threshold,
        while removing the neighbor with the lowest utility for that node

        TODO: Check if utility == 0 is ever possible and doesn't get removed
        """
        adjacency = super().get_adjacency()
        utility = self.get_utility()[nodes]
        total_cost_per_node = (self.cost * adjacency).sum(axis=1)[nodes]
        cost_to_remove = total_cost_per_node - self.cost_threshold
        neighbor_costs = (self.cost * adjacency)[nodes]
        removable_neighbors = neighbor_costs >= cost_to_remove[..., None]
        # Find rows that cant reach the threshold by removing one node
        rows_zero_removables = np.all((removable_neighbors == 0), axis=1)
        # Re add all neighbors of the nodes that didn't select any nodes
        removable_neighbors = (removable_neighbors + (adjacency[nodes] * rows_zero_removables[..., None])) == 1

        removable_neighbors_utility = utility * removable_neighbors
        # Mask 0s and pick the min utility for each node
        neighbors_to_remove = np.argmin(np.ma.masked_where(removable_neighbors_utility == 0, removable_neighbors_utility), axis=1)

        self.remove_node_adjacency_and_utility(nodes, neighbors_to_remove)

    def remove_node_adjacency_and_utility(self, nodes, neighbors):
        self.new_adjacency[nodes, neighbors] = 0
        self.new_adjacency[neighbors, nodes] = 0
        self.new_edge_utility[nodes, neighbors] = 0
        self.new_edge_utility[neighbors, nodes] = 0
        self.graph_changed = True

    def add_node_connections(self, nodes):
        eligible_connections = self.get_sampled_nodes()[nodes]
        # If there are no ellgible connections at all
        if (eligible_connections == 0).all():
            return

        adjacency = super().get_adjacency()
        utility = self.get_utility()[nodes]

        total_cost_per_node = (self.cost * adjacency).sum(axis=1)
        cost_remaining = self.cost_threshold - total_cost_per_node[nodes]
        
        eligible_total_costs = (eligible_connections * total_cost_per_node)
        eligible_costs = self.cost[nodes] * eligible_connections

        addable_neighbors = (eligible_costs <= cost_remaining[..., None]) & (eligible_costs > 0) & (eligible_total_costs <= self.cost_threshold)
 
        # Find rows that can add nodes without exceeding the cost threshold
        rows_addable = np.all((addable_neighbors == 0), axis=1) == False

        addable_neighbors_utility = utility * addable_neighbors

        # Pick the eligible nodes with the maximum utility
        neighbors_to_add = np.argmax(addable_neighbors_utility, axis=1)

        self.add_node_adjacency(nodes[rows_addable], neighbors_to_add[rows_addable])

    def add_node_adjacency(self, nodes, neighbors):
        self.new_adjacency[nodes, neighbors] = 1
        self.new_adjacency[neighbors, nodes] = 1
        self.new_edge_utility[nodes, neighbors] = 1
        self.new_edge_utility[neighbors, nodes] = 1
        self.graph_changed = True

    def get_sampled_nodes(self):
        """
        Create a matrix of eligble nodes per node to which a connection can be formed,
        Every row indicates a node source node and every column a node that a connection can be formed with
        This function does not take any cost or utility into account

        The definition of eligible nodes depends on the set SampleMethod enum
        """
        if self.sample_method == SampleMethod.ALL:
            eligible_nodes = self.get_inverted_adjacency_matrix()
        elif self.sample_method == SampleMethod.NEIGHBORS_OF_NEIGHBORS:
            eligible_nodes = self.get_eligible_neighbors_of_neighbors()
        elif self.sample_method == SampleMethod.CUSTOM:
            eligible_nodes = self.custom_sample_function()
        return eligible_nodes

    def get_inverted_adjacency_matrix(self):
        """
        Returns an inverted adjacency matrix, where every 1 means that there is no connection between nodes
        """
        adjacency = super().get_adjacency()
        inverted_adjacency = 1 - adjacency
        np.fill_diagonal(inverted_adjacency, 0)
        return inverted_adjacency

    def get_eligible_neighbors_of_neighbors(self):
        """
        Returns a matrix where a row is a node,
        and a 1 means that the column node is a neighbor of a neighbor of the row node
        """
        return super().get_neighbors_neighbors_adjacency_matrix()

    """
    Helper functions
    """

    def get_utility(self):
        return self.edge_utility

    def get_nodes_utility(self, nodes):
        return self.edge_utility[nodes]

    def get_previous_nodes_utility(self, n):
        """
        Get all the nodes' utility from the n'th previous saved iteration
        """
        available_iterations = list(self.simulation_output['utility'].keys())
        return self.simulation_output['utility'][available_iterations[-n - 1]]

    """
    Memory and writing functions
    """

    def prepare_output(self, n):
        super().prepare_output(n)
        n_nodes = len(self.graph.nodes)
        if self.config.utility_memory_config.save_disk:
            with open(self.config.utility_memory_config.path, 'w') as f:
                f.write(self.config.utility_memory_config.get_description_string(n, n_nodes, n_nodes))

    def write_simulation_step(self):
        super().write_simulation_step()
        n_nodes = len(self.graph.nodes)
        if self.config.utility_memory_config.save_disk and self.current_iteration % self.config.utility_memory_config.save_interval == 0:
            self.write_utility_iteration(n_nodes)

    def write_utility_iteration(self, n_nodes):
        with open(self.config.utility_memory_config.path, 'a') as f:
            f.write('# Iteration {0} - ({1}, {2})\n'.format(self.current_iteration, n_nodes, n_nodes))
            np.savetxt(f, self.edge_utility)

    def store_simulation_step(self):
        super().store_simulation_step()
        self.store_utility_iteration()

    def store_utility_iteration(self):
        if self.config.utility_memory_config.memory_size != -1 and self.current_iteration % self.config.utility_memory_config.memory_interval == 0:
            self.simulation_output['utility'][self.current_iteration] = copy.deepcopy(self.edge_utility)
        if self.config.utility_memory_config.memory_size > 0 and len(self.simulation_output) > self.config.utility_memory_config.memory_size:
            self.simulation_output['utility'] = {}
