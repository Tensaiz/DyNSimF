import os
from ast import literal_eval as make_tuple

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class VisualizationConfigurationException(Exception):
    """Configuration Exception"""


class VisualizationConfiguration(object):
    """
    Configuration for the visualizer
    TODO: Validate attributes
    """
    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)
        self.validate()

    def validate(self):
        if 'initial_positions' not in self.__dict__:
            self.initial_positions = None
        if 'repeat' not in self.__dict__:
            self.repeat = False
        if 'save_fps' not in self.__dict__:
            self.save_fps = 5

class Visualizer(object):
    """
    Visualizer class handling animations and plotting
    """
    def __init__(self, config, graph, state_map, model_output):
        self.config = config
        self.graph = graph
        self.state_map = state_map
        self.states = model_output['states']
        self.utilities = model_output['utility']
        self.adjacencies = model_output['adjacency']
        self.create_locations()
        self.max_iteration = self.get_total_iterations()

    def create_locations(self):
        print('Creating locations for adjacency graphs...')
        if len(self.adjacencies.values()) > 0:
            self.assign_dynamic_graph_values()
        else:
            self.assign_static_graph_values()

    def assign_dynamic_graph_values(self):
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
        # If it is the first iteration and initial positions have been given, use them
        if iteration == list(self.adjacencies.keys())[0] and self.config.initial_positions:
            self.locations[iteration] = self.config.initial_positions
            self.graphs[iteration] = nx.convert_matrix.from_numpy_array(adjacency_matrix)
        else:
            # Otherwise use the previous graph's calculated positions as initial positions if there are
            initial_positions = self.get_initial_positions(last_index)
            graph, locations = self.create_adjacency_node_locations(adjacency_matrix, initial_positions)
            self.locations[iteration] = locations
            self.graphs[iteration] = graph

    def get_initial_positions(self, last_index):
        if last_index in self.locations:
            initial_positions = self.locations[last_index]
        else:
            initial_positions = None
        return initial_positions

    def create_adjacency_node_locations(self, adjacency_matrix, initial_positions):
        # Adjacency matrix to graph
        graph = nx.convert_matrix.from_numpy_array(adjacency_matrix)
        positions = self.create_layout(graph, initial_positions)
        return (graph, positions)

    def create_layout(self, graph, initial_positions=None):
        if 'layout' in self.config.__dict__:
            if self.config.layout == 'fr':
                import pyintergraph
                Graph = pyintergraph.InterGraph.from_networkx(graph)
                G = Graph.to_igraph()
                positions = G.layout_fruchterman_reingold(niter=500)
            else:
                if 'layout_params' in self.config.__dict__:
                    positions = self.config.layout(graph,
                                             **self.config.layout_params)
                else:
                    positions = self.config.layout(graph)
        else:
            positions = nx.drawing.spring_layout(graph, pos=initial_positions, k=0.15)
        return positions

    def create_graph_node_locations(self):
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
        """
        Reads in saved states to disk and returns a numpy array

        Note: Has to be reworked
        """
        lines = open(path, 'r').readlines()
        dimensions = make_tuple(lines[1][1:])
        return np.loadtxt(path).reshape(dimensions)

    def visualize(self, vis_type):
        visualizations = {
            'animation': self.animation
        }
        visualizations[vis_type]()

    def setup_animation(self):
        state_names = list(self.state_map.keys())
        n_states = len(state_names)

        node_colors = self.get_node_colors()

        fig = plt.figure(figsize=(10, 9), constrained_layout=True)
        gs = fig.add_gridspec(6, n_states)

        network = fig.add_subplot(gs[:-1, :])

        axis = []

        for i in range(n_states):
            ax = fig.add_subplot(gs[-1, i])
            ax.set_title(state_names[i])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            axis.append(ax)

        n = int(self.max_iteration / self.config.plot_interval)

        cm = plt.cm.get_cmap(self.config.color_scale)
        vmin = self.config.variable_limits[self.config.plot_variable][0]
        vmax = self.config.variable_limits[self.config.plot_variable][1]
        colors = cm(np.linspace(0, 1, 25))
        return state_names, n_states, node_colors, fig, gs, network, axis, n, cm, vmin, vmax, colors

    def animation(self):
        state_names, n_states, node_colors, fig, gs, network, axis, n, cm, vmin, vmax, colors = \
            self.setup_animation()

        def animate(curr):
            index = curr * self.config.plot_interval
            state_index = self.get_last_index(self.states, index)

            network.clear()
            for i, ax in enumerate(axis):
                ax.clear()
                data = self.states[state_index][:, i]
                bc = ax.hist(data,
                             range=self.config.variable_limits[state_names[i]],
                             density=1, bins=25, edgecolor='black')[2]
                for j, e in enumerate(bc):
                    e.set_facecolor(colors[j])
                ax.set_title(state_names[i])

            if self.locations:
                locations_index = self.get_last_index(self.adjacencies, index)
                # self.graph = nx.convert_matrix.from_numpy_array(self.adjacencies[locations_index])
                graph = self.graphs[locations_index]
                pos = self.locations[locations_index]
            else:
                pos = self.static_locations
                graph = self.graph
            nx.draw_networkx_edges(graph, pos,
                                   alpha=0.2, ax=network)
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

        if 'plot_output' in self.config.__dict__:
            self.save_plot(ani)

    def save_plot(self, simulation):
        """
        Save the plot to a file,
        specified in plot_output in the visualization configuration
        The file is generated using the writer from the pillow library

        :param simulation: Output of matplotlib animation.FuncAnimation
        """
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
        iterations = list(self.states.keys())
        node_colors = []
        for i in range(len(self.states.keys())):
            if i % self.config.plot_interval == 0:
                node_colors.append(
                    [self.states[iterations[i]][node, self.state_map[self.config.plot_variable]]
                    for node in self.graph.nodes]
                )
        return node_colors

    def get_last_index(self, variable, index):
        iterations = list(variable.keys())
        if index < iterations[0]:
            return iterations[0]
        while index not in iterations:
            index -= 1
        return index

    def get_total_iterations(self):
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
