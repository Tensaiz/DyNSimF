import os
from ast import literal_eval as make_tuple
import types

from dynsimf.models.helpers.ConfigValidator import ConfigValidator

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class VisualizationConfiguration(object):
    '''
    Configuration for the visualizer

    :var int plot_interval: An integer indicating the interval between iterations to store a new plot, defaults to 1
    :var dict initial_positions: A dictionary with 
        (int) node (key): tuple((x_pos (float), y_pos (float))) (value) pairs that indicate the initial positions,
        defaults to `None`
    :var dict fixed_positions: A dictionary with 
        (int) node (key): tuple((x_pos (float), y_pos (float))) (value) pairs that indicate stationary positions of the nodes,
        defaults to `None`
    :var str plot_variable: A string indicating the main variable to plot as the node color, defaults to `None`
    :var str color_scale: A string indicating the matplotlib color scale to use, defaults to `Reds`
    :var bool show_plot: Inidicates whether the animated plot should be shown after the simulation, defaults to `True`
    :var bool repeat: Indicates whether the animated plot should repeat itself after fully playing, defaults to `True`
    :var str plot_output: The path + name + extension of where the output of the animation should be saved,
        if it is not set, the plot will not be saved, defaults to `None`
    :var int save_fps: The frames per second the saved animation should have, defaults to 5
    :var str plot_title: The title of the plot, defaults to `Network visualization`
    :var layout: The layout of the network used for the visualization, 
        this can be a networkx layout function, a custom function, or a string indicating a built-in function, like `fr`.
        This will then use the `layout_fruchterman_reingold` from the igraph package. If a custom function is used, 
        it should return a dictionary where they keys are nodes and the values are tuples with their x and y positions as floats.
        Defaults to `None`
    :vartype layout: str or function
    :var dict layout_params: A dictionary that is unpacked when calling the layout function if a custom layout function is set. 
        The keys should correspond with the parameters of the function. Defaults to `None`
    :var float edge_alpha: The alpha of the edges between nodes in the plot, defaults to `0.2`
    :var str edge_values: A string indicating which `edge_values` variable from the model should be used to visualize the edge colors,
        defaults to `None` 
    :var bool directed: Indicates whether the network plot should have directed edges, defaults to `True`
    '''
    def __init__(self, config):
        '''
        Initialise the visualization config

        :param dict config: Dictionary containing the values for the members of the configuration class,
            When certain keys/value pairs are not provided, default values are used
        '''
        self.set_config(config)
        self.validate()

    def set_config(self, config):
        '''
        Set the config, its default values and validate it after

        :param dict config: Dictionary containing the values for the members of the configuration class,
            When certain keys/value pairs are not provided, default values are used
        '''
        self.config = config
        self.plot_interval = config['plot_interval'] if 'plot_interval' in config else 1
        self.initial_positions = config['initial_positions'] if 'initial_positions' in config else None
        self.fixed_positions = config['fixed_positions'] if 'fixed_positions' in config else None
        self.plot_variable = config['plot_variable'] if 'plot_variable' in config else None
        self.color_scale = config['color_scale'] if 'color_scale' in config else 'Reds'
        self.show_plot = config['show_plot'] if 'show_plot' in config else True
        self.repeat = config['repeat'] if 'repeat' in config else True
        self.plot_output = config['plot_output'] if 'plot_output' in config else None
        self.save_fps = config['save_fps'] if 'save_fps' in config else 5
        self.plot_title = config['plot_title'] if 'plot_title' in config else 'Network visualization'
        self.layout = config['layout'] if 'layout' in config else None
        self.layout_params = config['layout_params'] if 'layout_params' in config else None
        self.edge_alpha = config['edge_alpha'] if 'edge_alpha' in config else 0.2
        self.edge_values = config['edge_values'] if 'edge_values' in config else None
        self.directed = config['directed'] if 'directed' in config else True
        self.histogram_states = config['histogram_states'] if 'histogram_states' in config else None

        if 'variable_limits' not in config:
            self.variable_limits = {state: [-1, 1] for state in config['state_names']}
        else:
            for state in config['state_names']:
                if state not in config['variable_limits']:
                    config['variable_limits'][state] = [-1, 1]
            config['variable_limits']['utility'] = [0, 1]
            self.variable_limits = config['variable_limits']

    def validate(self):
        '''
        Validate the config using the `ConfigValidator` class.

        :raises ValueError: if a class member does not match its expected type or range, 
            or if is not optional and has not been set
        '''
        ConfigValidator.validate('plot_interval', self.plot_interval, int, variable_range=(0, ))
        ConfigValidator.validate('initial_positions', self.initial_positions, dict, optional=True)
        ConfigValidator.validate('fixed_positions', self.fixed_positions, dict, optional=True)
        ConfigValidator.validate('plot_variable', self.plot_variable, str)
        if self.plot_variable not in self.config['state_names'] and self.plot_variable != 'utility':
            raise ValueError('The plot variable ' + self.plot_variable + \
                ' does not exist in the model\'s states: ' + str(self.config['state_names']) +\
                ' and it is not equal to utility')
        ConfigValidator.validate('color_scale', self.color_scale, str)
        ConfigValidator.validate('show_plot', self.show_plot, bool)
        ConfigValidator.validate('repeat', self.repeat, bool)
        ConfigValidator.validate('plot_output', self.plot_output, str, optional=True)
        ConfigValidator.validate('save_fps', self.save_fps, int)
        ConfigValidator.validate('plot_title', self.plot_title, str)
        ConfigValidator.validate('variable_limits', self.variable_limits, dict)
        ConfigValidator.check_optionality('layout', self.layout, True)
        if self.layout:
            ConfigValidator.check_types('layout', self.layout, [str, types.FunctionType])
        ConfigValidator.validate('layout_params', self.layout_params, dict, optional=True)
        ConfigValidator.validate('edge_alpha', self.edge_alpha, float, variable_range=(0, 1))
        ConfigValidator.validate('edge_values', self.edge_values, str, optional=True)
        ConfigValidator.validate('directed', self.directed, bool)
        ConfigValidator.validate('histogram_states', self.histogram_states, list, optional=True)

class Visualizer(object):
    '''
    Visualizer class handling animations and plotting

    :var VisualizationConfiguration config: The config class containing all configuration values
    :var networkx.classes.graph.Graph graph: The networkx graph to plot
    :var dict state_map: A dictionary mapping the state names (str) to the index of the state column in the 2d states array 
    :var dict edge_values_map: A dictionary mapping the edge value names (str) to the index of the edge values arrays
    :var dict states: A dictionary where the key is an iteration (int) and the values are a 2d array with columns as state values and rows as nodes
    :var dict adjacencies: A dictionary where the key is an iteration (int) and the values are network adjacency matrices
    :var dict edge_values: A dictionary where the key is an iteration (int) and the values are network edge values
    :var dict utilities: A dictionary where the key is an iteration (int) and the values are utility values per edge
    :var int max_iteration: The max amount of iterations that are available
    '''
    def __init__(self, config, model_input):
        '''
        Initialize the visualizer object by assigning the class members and then generating locations for the network for each iteration
        '''
        graph, state_map, model_output, edge_values_map = model_input
        self.config = config
        self.graph = graph
        self.state_map = state_map
        self.edge_values_map = edge_values_map
        self.states = model_output['states']
        self.adjacencies = model_output['adjacency']
        self.edge_values = model_output['edge_values']
        if 'utility' in model_output:
            self.utilities = model_output['utility']
        else:
            self.utilities = {}
        self.create_locations()
        self.max_iteration = self.get_total_iterations()
        self.histogram_states = list(self.state_map.keys()) if self.config.histogram_states is None else self.config.histogram_states

    def create_locations(self):
        '''
        Generate locations for all the adjacency graphs for each iteration
        '''
        print('Creating locations for adjacency graphs...')
        if len(self.adjacencies.values()) > 0:
            self.assign_dynamic_graph_values()
        else:
            self.assign_static_graph_values()

    def assign_dynamic_graph_values(self):
        '''
        For each iteration and adjacency matrix, generate new locations or use the previous ones if the graph hasn't changed
        '''
        self.locations = {}
        self.graphs = {}
        for iteration, adjacency_matrix in self.adjacencies.items():
            last_index = self.get_last_index(self.adjacencies, iteration - 1)
            prev_adj = self.adjacencies[last_index]
            # If the adjacency matrix hasn't changed and the locations are already calculated
            # Use the previous locations and graph
            if (prev_adj == adjacency_matrix).all() and last_index in self.locations:
                self.locations[iteration] = self.locations[last_index]
                self.graphs[iteration] = self.graphs[last_index]
            # Otherwise create new positions and graph
            else:
                self.create_new_positions_graph(iteration, adjacency_matrix, last_index)

    def create_new_positions_graph(self, iteration, adjacency_matrix, last_index):
        '''
        Create new graphs: if it is the first iteration and initial positions have been given, use them,
        otherwise use the previous graph's calculated positions as initial positions if there are any

        :param int iteration: The iteration to store and create the positions for
        :param numpy.ndarray adjacency_matrix: The adjacency matrix that the positions should be generated for
        :param int last_index: The last iteration that the locations were generated for
        '''
        # If it is the first iteration and initial positions have been given, use them
        if iteration == list(self.adjacencies.keys())[0] and self.config.initial_positions:
            self.locations[iteration] = self.config.initial_positions
            if self.config.directed:
                self.graphs[iteration] = nx.convert_matrix.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
            else:
                self.graphs[iteration] = nx.convert_matrix.from_numpy_array(adjacency_matrix)
        else:
            # Otherwise use the previous graph's calculated positions as initial positions if there are
            initial_positions = self.get_initial_positions(last_index)
            graph, locations = self.create_adjacency_node_locations(adjacency_matrix, initial_positions)
            self.locations[iteration] = locations
            self.graphs[iteration] = graph

    def get_initial_positions(self, last_index):
        '''
        Check if `last_index` has already any locations generated, if so use it as initial position for the next location generation

        :return: Initial positions to use for the next location generation
        :rtype: dict or None
        '''
        if last_index in self.locations:
            initial_positions = self.locations[last_index]
        else:
            initial_positions = None
        return initial_positions

    def create_adjacency_node_locations(self, adjacency_matrix, initial_positions):
        '''
        Create a networkx graph from an adjacency matrix and return the generated locations of the nodes

        :param numpy.ndarray adjacency_matrix: The adjacency matrix to turn into a networkx graph
        :param initial_positions: The initial positions to use when generating new positions
        :type initial_positions: Dict or None 
        :return: A `tuple` containing the new networkx graph and the corresponding node positions
        :rtype: `tuple`
        '''
        # Adjacency matrix to graph
        if self.config.directed:
            graph = nx.convert_matrix.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
        else:
            graph = nx.convert_matrix.from_numpy_array(adjacency_matrix)
        if self.config.fixed_positions:
            positions = self.config.fixed_positions
        else:
            positions = self.create_layout(graph, initial_positions)
        return (graph, positions)

    def create_layout(self, graph, initial_positions=None):
        '''
        Generate new positions for the nodes in a given graph, using initial positions if given

        :param networkx.classes.graph.Graph graph: The graph with the nodes to generate the locations for
        :param initial_positions: Optional initial positions to use when creating the node locations
        :type initial_positions: Dict, optional
        :return: A dictionary with nodes as keys and tuples of form (x_pos, y_pos) as values
        :rtype: dict
        '''
        if self.config.layout:
            if self.config.layout == 'fr':
                import pyintergraph
                Graph = pyintergraph.InterGraph.from_networkx(graph)
                G = Graph.to_igraph()
                positions = G.layout_fruchterman_reingold(niter=500)
            else:
                if self.config.layout_params:
                    positions = self.config.layout(graph,
                                             **self.config.layout_params)
                else:
                    positions = self.config.layout(graph)
        else:
            positions = nx.drawing.spring_layout(graph, pos=initial_positions, k=0.25, iterations=5)
            # positions = nx.drawing.kamada_kawai_layout(graph, pos=initial_positions)
        return positions

    def create_graph_node_locations(self):
        '''
        Create locations of the graph for graph is static positions is set, use intitial positions if set
        '''
        if self.config.initial_positions:
            self.static_locations = self.config.initial_positions
        elif 'pos' in self.graph.nodes[0].keys():
            self.static_locations = nx.get_node_attributes(self.graph, 'pos')
        else:
            self.static_locations = self.create_layout(self.graph)

    def assign_static_graph_values(self):
        self.locations = None
        self.graphs = None
        self.create_graph_node_locations()

    @staticmethod
    def read_states_from_file(path):
        '''
        Reads in saved states to disk and returns a numpy array

        Note: Has to be reworked
        
        :param str path: The path to read the states from
        '''
        lines = open(path, 'r').readlines()
        dimensions = make_tuple(lines[1][1:])
        return np.loadtxt(path).reshape(dimensions)

    def visualize(self, vis_type):
        '''
        Navigate to the appropriate visualisation function
        '''
        visualizations = {
            'animation': self.animation
        }
        visualizations[vis_type]()

    def setup_animation(self):
        '''
        Setup the animation by creating all the necessary elements for the plot

        :return: tuple containing all necessary plot elements:
            state_names: names of all the states to display, 
            n_states: integer with all the states, 
            node_colors: list will all the colors of the nodes for each iteration, 
            fig: matplotlib figure, 
            gs: figure gridspec, 
            network: subplot to plot the network in, 
            axis: list of axes to display the barplots, 
            n: amount of plots, 
            cm: colormap to use, 
            vmin: minimum value of the node colors, 
            vmax: maximum value of the node colors, 
            colors: 25 values between 0 and 1 on the colormap
        '''
        n_states = len(self.histogram_states)

        node_colors = self.get_node_colors()

        fig = plt.figure(figsize=(10, 9), constrained_layout=True)
        gs = fig.add_gridspec(6, n_states)

        network = fig.add_subplot(gs[:-1, :])

        axis = []

        for i in range(n_states):
            ax = fig.add_subplot(gs[-1, i])
            ax.set_title(self.histogram_states[i])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            axis.append(ax)

        n = int(self.max_iteration / self.config.plot_interval)

        cm = plt.cm.get_cmap(self.config.color_scale)
        vmin = self.config.variable_limits[self.config.plot_variable][0]
        vmax = self.config.variable_limits[self.config.plot_variable][1]
        colors = cm(np.linspace(0, 1, 25))
        return n_states, node_colors, fig, gs, network, axis, n, cm, vmin, vmax, colors

    def animation(self):
        '''
        This function displays and animates the visualization.

        The plot is being cleared and redrawn every plot. If configured, the plot will be shown and saved.
        '''
        n_states, node_colors, fig, gs, network, axis, n, cm, vmin, vmax, colors = \
            self.setup_animation()

        def animate(curr):
            index = curr * self.config.plot_interval
            state_index = self.get_last_index(self.states, index)

            network.clear()
            for i, ax in enumerate(axis):
                ax.clear()

                histogram_index = self.state_map[self.histogram_states[i]]
                data = self.states[state_index][:, histogram_index]
                bc = ax.hist(data,
                             range=self.config.variable_limits[self.histogram_states[i]],
                             density=1, bins=25, edgecolor='black')[2]
                for j, e in enumerate(bc):
                    e.set_facecolor(colors[j])
                ax.set_title(self.histogram_states[i])

            edge_color = None
            if self.locations:
                locations_index = self.get_last_index(self.adjacencies, index)
                # self.graph = nx.convert_matrix.from_numpy_array(self.adjacencies[locations_index])
                graph = self.graphs[locations_index]
                pos = self.locations[locations_index]
                if self.config.edge_values:
                    if locations_index in self.edge_values:
                        current_edge_values = self.edge_values[locations_index][self.edge_values_map[self.config.edge_values]]
                        edge_color = current_edge_values.flatten()[self.adjacencies[locations_index].flatten().nonzero()[0]]
                else:
                    if locations_index in self.utilities:
                        edge_color = self.utilities[locations_index].flatten()[self.adjacencies[locations_index].flatten().nonzero()[0]]
            else:
                pos = self.static_locations
                graph = self.graph

            nx.draw_networkx_edges(graph, pos,
                                   alpha=self.config.edge_alpha, ax=network, connectionstyle='arc3, rad=0.1', edge_color=edge_color, edge_cmap=cm, edge_vmin=0, edge_vmax=1)

            nc = nx.draw_networkx_nodes(graph, pos,
                                        nodelist=graph.nodes,
                                        node_color=node_colors[curr],
                                        vmin=vmin, vmax=vmax,
                                        cmap=cm, node_size=50,
                                        ax=network)
            nc.set_edgecolor('black')
            network.get_xaxis().set_ticks([])
            network.get_yaxis().set_ticks([])
            network.set_title('Iteration: ' + str(index))

        ani = animation.FuncAnimation(fig, animate, n, interval=200,
                                      repeat=self.config.repeat, blit=False)

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=network)
        fig.suptitle(self.config.plot_title, fontsize=16)

        if self.config.show_plot:
            plt.show()

        if self.config.plot_output:
            self.save_plot(ani)

    def save_plot(self, simulation):
        '''
        Save the plot to a file,
        specified in plot_output in the visualization configuration
        The file is generated using the writer from the pillow library

        :param simulation: Output of matplotlib animation.FuncAnimation
        '''
        print('Saving plot at: ' + self.config.plot_output + ' ...')
        split = self.config.plot_output.split('/')
        file_name = split[-1]
        file_path = self.config.plot_output.replace(file_name, '')
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        from PIL import Image
        writergif = animation.PillowWriter(fps=self.config.save_fps)
        simulation.save(self.config.plot_output, writer=writergif)
        print('Saved: ' + self.config.plot_output)

    def get_node_colors(self):
        '''
        Helper function to create a list of floats that indicate the colors of each node.

        If utility is selected, the values are taken from the utility arrays, otherwise the correct state is chosen

        :return: `list` of lists where each list corresponds to a list of floats that indicate the color for each node
        :rtype: list
        '''
        iterations = list(self.states.keys())
        node_colors = []
        if self.config.plot_variable == 'utility':
            for i in list(self.utilities.keys()):
                if i % self.config.plot_interval == 0:
                    node_colors.append(
                        (self.utilities[i] * self.adjacencies[i]).sum(axis=1)
                    )
        else:
            for i in range(len(self.states.keys())):
                if i % self.config.plot_interval == 0:
                    node_colors.append(
                        [self.states[iterations[i]][node, self.state_map[self.config.plot_variable]]
                        for node in self.graph.nodes]
                    )
        return node_colors

    def get_last_index(self, variable, index):
        '''
        Find the key `index` in the dictionary `variable`. 
        If `index` does not exist, subtract 1 from `index` and try again.
        Keep repeating this, until a value is found.

        :param dict variable: The dictionary to find the key for
        :param int index: The index to find the key in the dict for
        :return: The found key in the dictionary that matches the `index` parameter.
        :rtype: int
        '''
        iterations = list(variable.keys())
        if index < iterations[0]:
            return iterations[0]
        while index not in iterations:
            index -= 1
        return index

    def get_total_iterations(self):
        ''' 
        Get the total amount of iterations that can be used for the visualization

        :return: The total amount of iterations
        :rtype: int
        '''
        visualizables = [
            self.states,
            self.adjacencies,
            self.utilities
        ]
        total = 0
        for visualizable in visualizables:
            keys = list(visualizable.keys())
            if len(keys) > 0:
                max_iteration = keys[-1]
                if max_iteration > total:
                    total = max_iteration
        return total
