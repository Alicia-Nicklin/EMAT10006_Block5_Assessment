####################################################################
#                                                                  #
#              EMAT10006 Further Computer Programming  2024        #
#                                                                  #
#                      FCP Summative Assessment                    #
#                                                                  #
#                          Archie Miller                           #
#                           Tommy Newman                           #
#                          Alicia Nicklin                          #
#                            Eric Ogden                            #
#                                                                  #
####################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import argparse
import random

class Node:

    def __init__(self, value, number, connections=None):
        self.index = number
        self.connections = connections
        self.value = value


class Network:

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def get_mean_degree(self):

        # Your code  for task 3 goes here
        assert(0)

    def get_mean_clustering(self):

        # Your code for task 3 goes here
        assert(0)

    def get_mean_path_length(self):

        # Your code for task 3 goes here
        assert(0)

    def make_random_network(self, N, connection_probability):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        '''

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_ring_network(self, N, neighbour_range=2):

        # Your code  for task 4 goes here
        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))
        for (index, node) in enumerate(self.nodes):
                for neighbour_index in range(index + 1, index + 1 + neighbour_range):
                    if neighbour_index >= N:
                        neighbour_index = neighbour_index -N
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1


    def make_small_world_network(self, N, re_wire_prob=0.2):

        # Your code for task 4 goes here
        neighbour_range = 2
        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))
        for (index, node) in enumerate(self.nodes):
            if np.random.random() < re_wire_prob:
                # Select a random node
                while True:
                    random_node=random.randint(0, N-1)
                    if node.connections[index] == 0:
                        node.connections[random_node] = 1
                        self.nodes[random_node].connections[index] = 1
                        break
            else:
                for neighbour_index in range(index + 1, index + 1 + neighbour_range):
                    if neighbour_index >= N:
                        neighbour_index = neighbour_index - N
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        # Rather than just set "0.3 * num_nodes" as the radius of each small circle
        # lets calculate it using basic trig

        each_small_circle_radius = 0.3 * num_nodes
        if (num_nodes>2):
            each_arc_angle = 360 / num_nodes
            step1 = network_radius * np.sin(np.deg2rad(each_arc_angle))
            step2 = 180-each_arc_angle
            step3 = step2/2
            step4 = 2 * np.sin(np.deg2rad(step3))
            each_small_circle_radius = (step1 / step4) - 2  # This -2 is just to put a tiny bit of space between

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), each_small_circle_radius, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')
        plt.autoscale()
        plt.show()

def test_networks():
    # Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number - 1) % num_nodes] = 1
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert (network.get_mean_degree() == 2), network.get_mean_degree()
    assert (network.get_clustering() == 0), network.get_clustering()
    assert (network.get_path_length() == 2.777777777777778), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert (network.get_mean_degree() == 1), network.get_mean_degree()
    assert (network.get_clustering() == 0), network.get_clustering()
    assert (network.get_path_length() == 5), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert (network.get_mean_degree() == num_nodes - 1), network.get_mean_degree()
    assert (network.get_clustering() == 1), network.get_clustering()
    assert (network.get_path_length() == 1), network.get_path_length()

    print("All tests passed")


'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''


def calculate_agreement(population, row, col, external=0.0):
    '''
    This function should return the *change* in agreement that would result if the cell at (row, col) was to flip it's value
    Inputs: population (numpy array)
            row (int)
            col (int)
            external (float)
    Returns:
            change_in_agreement (float)
    '''

    # Your code for task 1 goes here
    assert (0)

    return np.random * population


def create_ising_population():
    population = np.random.rand(10, 10)
    for i in range(10):
        for j in range(10):
            if population[i][j] <= 0.5:
                population[i][j] = -1
            else:
                population[i][j] = 1
    return population

def ising_step(population, external=0.0):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
            external (float) - optional - the magnitude of any external "pull" on opinion
    '''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, external=0.0)

    if agreement < 0:
        population[row, col] *= -1


    # Your code for task 1 goes here

def plot_ising(im, population):
    '''
    This function will display a plot of the Ising model
    '''

    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)


def test_ising():
    '''
    This function will test the calculate_agreement function in the Ising model
    '''

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1) == 4), "Test 1"

    population[1, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -4), "Test 2"

    population[0, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -2), "Test 3"

    population[1, 0] = 1.
    assert (calculate_agreement(population, 1, 1) == 0), "Test 4"

    population[2, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == 2), "Test 5"

    population[1, 2] = 1.
    assert (calculate_agreement(population, 1, 1) == 4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
    assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
    assert (calculate_agreement(population, 1, 1, 10) == 14), "Test 9"
    assert (calculate_agreement(population, 1, 1, -10) == -6), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''


def defuant_main(beta=0.2, threshold=0.2):

    # Your code for task 2 goes here
    assert (0)

def test_defuant():

    # Your code for task 2 goes here
    assert (0)

'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''


def main():

    parser = argparse.ArgumentParser()

    # Task 1 command line parameters
    parser.add_argument("-ising_model", type=int, help="Ising model with default parameters")
    parser.add_argument("-external", type=float, default=0.0,
                        help="Ising external value. Defaults to 0")
    parser.add_argument("-alpha", type=int, default=1,
                        help="Ising temperature value. Defaults to 1")
    parser.add_argument("-test_ising", type=int, help="Run Ising tests")


    # Task 2 command line parameters
    parser.add_argument("-defuant", type=int, help="Defuant model with default parameters")
    parser.add_argument("-beta", type=float, default=0.2,
                        help="Defuant beta value. Defaults to 0.2")
    parser.add_argument("-threshold", type=float, default=0.2,
                        help="Defuant threshold value. Defaults to 0.2")
    parser.add_argument("-test_defuant", type=int, help="Run defuant tests")


    # Task 3 command line parameters
    parser.add_argument("-network", type=int, help="Create a random network, size of n")
    parser.add_argument("-test_network", type=int, help="Run network tests")


    # Task 4 command line parameters
    parser.add_argument("-random_network", type=int, help="Create a random network, size of n")
    parser.add_argument("-connection_probability", type=float, default=0.3,
                        help="Connection probability. Defaults to 0.3")
    parser.add_argument("-ring_network", type=int, help="Create a ring network with a range of 1 and a size of n")
    parser.add_argument("-range", type=int, default=2, help="Network range. Defaults to 2")
    parser.add_argument("-small_world", type=int, help="Create a small-worlds network with default parameters, size n")
    parser.add_argument("-re_wire", type=float, default=0.2, help="Re-wire probability. Defaults to 0.2")

    args = parser.parse_args()

    # Task 1 calls
    if args.test_ising:
        test_ising()
    if args.ising_model:
        ising_main(create_ising_population(), args.alpha, args.external)


    # Task 2 calls
    if args.defuant:
        defuant_main(args.beta, args.threshold)
    if args.test_defuant:
        test_defuant()


    # Task 3 calls
    if args.network:
        network = Network()
        print("Mean degree: " + str(network.mean_degree()))
        print("Average path length: " + str(network.average_path_length()))
        print("Clustering co - efficient:" + str(network.clustering_co_efficient()))
        network.plot()
    if args.test_network:
        test_networks()


    # Task 4 calls
    if args.random_network:
        network = Network()
        network.make_random_network(args.random_network, args.connection_probability)
        network.plot()
    if args.ring_network:
        network = Network()
        network.make_ring_network(args.ring_network, args.range)
        network.plot()
    if args.small_world:
        network = Network()
        network.make_small_world_network(args.small_world, args.re_wire)
        network.plot()

    # Task 5



if __name__ == "__main__":
    main()