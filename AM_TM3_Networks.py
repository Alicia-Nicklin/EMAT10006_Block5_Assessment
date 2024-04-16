
import numpy as np
import matplotlib.pyplot as plt

class Node:
    '''
    Class to represent a node in an undirected graph
    Each node has a floating point value and some neighbours
    Neighbours is a numpy array representing the row of the adjacency matrix that corresponds to the node
    '''

    def __init__(self, index, value, connections=[]):
        self.value = value
        self.connections = connections

    def get_neighbours(self):
        return np.where(np.array(self.connections) == 1)[0]

class Graph:
    def __init__(self, nodes, adjacency_matrix):
        self.nodes = []
        for (i, node) in enumerate(nodes):
            new_node = Node(i, node, adjacency_matrix[i])
            self.nodes.append(new_node)
        self.adjacency_matrix = adjacency_matrix

    def plot(self, fig=None, ax=None):
        if fig == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_axis_off()
        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])
        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)
            circle = plt.Circle((node_x, node_y), network_radius / 10, fill=False)
            ax.add_patch(circle)
            ax.text(node_x * 1.05, node_y * 1.05, node.value)
            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)
                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')
        plt.show()


def get_degree(self):
    i_degree = 0
    Node(value, number,adjk)

    return


def get_mean_degree(self):


def get_mean_clustering(self):
	#Your code for task 3 goes here

def get_mean_path_length(self):
	#Your code for task 3 goes here

'''

nodes = [5, 2, 4, 1]
connectivity = np.array([[0, 0, 1, 1],
                         [0, 0, 1, 1],
                         [1, 1, 0, 0],
                         [1, 1, 0, 0]])
graph = Graph(nodes, connectivity)
node = Node()
graph.plot()
