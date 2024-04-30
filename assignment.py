####################################################################
#                                                                  #
#              EMAT10006 Further Computer Programming  2024        #
#                                                                  #
#                      FCP Summative Assessment                    #
#                                                                  #
#                          Archie Miller                           #
#                           Tommy Newmanx                           #
#                          Alicia Nicklin                          #
#                            Eric Ogden                            #
#                                                                  #
####################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import argparse
import random
import sys

class Queue:
    def __init__(self):
        self.queue = []

    def push(self,item):
        self.queue.append(item)

    def pop(self):
        return self.queue.pop(0)

    def is_empty(self):
        return len(self.queue)==0




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
        '''
        Calculates the average of the degree of all nodes in the networkss
        '''

        count = 0 # initializes variable to sum number of nodes

        for node in self.nodes: # iterates through nodes in network
                count += sum(node.connections) # adds number of connections to count variable

        # calculates average by dividing count by number of nodes
        return count / len(self.nodes)

    def get_clustering(self):
        # Build up a list of cluster counts, one for each node
        clustering = []


        for node in self.nodes:

            triangle_count = 0  # initializes variable to sum number of triangles
            connections = node.connections
            neighbours_in = [i for i, j in enumerate(node.connections) if j == 1]  # lists the indices of neighbours
            n = sum(connections)  # calculates the number of neighbours
            possible_triangles = n * (n - 1) / 2    # calculates possible triangles from given formula

            if n <= 1:
                clustering.append(0)  # case where there are one or no neighbours so no triangles can be formed
                continue


            for n in neighbours_in:
                neighbour_n = self.nodes[n]  # gets connected node
                for each_original_neighbour in neighbours_in:
                    if neighbour_n.connections[each_original_neighbour] == 1: # check if node is a neighbour

                        triangle_count += 1


            # as every connection is counted in both directions the final value is divided by 2:
            true_triangles = triangle_count / 2

            clustering.append(true_triangles / possible_triangles)

        return clustering

    def get_mean_clustering(self):

        number_of_nodes = len(self.nodes)
        clustering = self.get_clustering()

        mean_clustering = sum(clustering) / number_of_nodes #calculates average

        return mean_clustering


    def bfs_path_length(self, start_node:Node, end_node : Node):
        '''
        A BFS using queues was used with added path length variable to find shortest path length
        '''

        queue = Queue()
        visited = []
        path_length = 0

        queue.push((start_node,0))
        visited.append(start_node)

        while not queue.is_empty():
            (node_to_check,current_path_length) = queue.pop()


            if node_to_check == end_node:
                path_length = current_path_length
                break

            for index,is_connected in enumerate(node_to_check.connections):
                neighbour = self.nodes[index]
                if is_connected and neighbour not in visited:
                    queue.push((neighbour, current_path_length + 1))
                    visited.append(neighbour)
                    neighbour.parent = node_to_check


        return path_length

    def get_path_length(self,index):
        lengths = []

        for node in self.nodes:
            if self.nodes[index] != node: #code only runs if it is not self nodes
                lengths.append((self.bfs_path_length(self.nodes[index], node)))

        return lengths
    def get_mean_path_length(self):
        """
        for this function we used an equation for mean path length
        """
        path_sum = 0
        for index, node in enumerate(self.nodes):
            path_lengths = self.get_path_length(node.index)
            path_sum += sum(path_lengths)

        mean_path_length = (1/((len(self.nodes)) * ((len(self.nodes) - 1))))  * path_sum

        return(mean_path_length)




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


    def make_default_network(self, N):
        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))


    def make_random_network(self, N, connection_probability):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        '''

        self.make_default_network(N)

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_ring_network(self, N, neighbour_range=2):

        # Your code  for task 4 goes

        self.make_default_network(N)

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

        self.make_default_network(N)

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
    This function should return the extent to which a cell agrees with its neighbours.
    Inputs: population (numpy array)
            row (int)
            col (int)
            external (float)
    Returns:
            change_in_agreement (float)
    '''
    n_rows, n_cols = population.shape
    agreement = 0
    if row > 0:
        agreement += population[row - 1, col] * population[row, col]
    if row < n_rows -1:
        agreement += population[row + 1, col] * population[row, col]
    if col > 0:
        agreement += population[row, col - 1] * population[row, col]
    if col < n_cols -1:
        agreement += population[row, col + 1] * population[row, col]

    change_in_agreement = agreement + external * population[row, col]    
    
    return change_in_agreement


def create_ising_population():
    population = np.random.rand(10, 10)
    for i in range(10):
        for j in range(10):
            if population[i, j] <= 0.5:
                population[i, j] = -1
            else:
                population[i, j] = 1
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

    change_in_agreement = calculate_agreement(population, row, col, external)

    #Probabiltity of flipping opinion
    flip_probability = np.exp(-change_in_agreement)
    
    if np.random.rand() < flip_probability:
        population[row, col] *= -1


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
    assert (calculate_agreement(population, 1, 1, 10) == -6), "Test 9"
    assert (calculate_agreement(population, 1, 1, -10) == 14), "Test 10"

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


def plot_histogram(population):
    '''
    This function plots the histogram from the final results after all iterations
    have been completed and sets the bin sizes accordingly and labels all axis
    and titles accordingly
    '''
    plt.figure()
    plt.hist(population, bins=np.arange(0, 1.1, 0.1), alpha=0.7)
    plt.xlabel("Opinions")
    plt.ylabel("Frequency")
    plt.title("Opinions Frequency Distribution")
    plt.grid(False)
    plt.show()

def plot_opinions(population_history):
    '''
    This function plots the scatter graph of each opinion at each timestep
    by taking the stored array and plotting each element at each iteration
    and labels the graphs accordingly
    '''
    plt.figure()
    iterations = len(population_history)
    num_individuals = len(population_history[0])
    for i in range(num_individuals):
        opinions = [population_history[j][i] for j in range(iterations)]
        plt.scatter(range(iterations), opinions, color='red')

    plt.xlabel("Time Step")
    plt.ylabel("Opinion")
    plt.title("Opinions Over Time")
    plt.grid(False)
    plt.show()

def population_indexes(population, population_size, i_index):
    '''
    This function randomly picks a person in the population and finds the opinion rating of that person and passes
    it back to the function when called upon
    '''

    i_b_index = (i_index - 1) % population_size
    i_a_index = (i_index + 1) % population_size

    i_b = population[i_b_index]
    i = population[i_index]
    i_a = population[i_a_index]

    return i_b, i, i_a, i_b_index, i_a_index

def update_opinions_after(i, i_a, threshold, beta):
    '''
    This function controls the neighbour after the selected person and updates
    that persons opinion using the equation given if their opinion is in the threshold
    '''
    if abs(i - i_a) < threshold:
        i_new = i + (beta * (i_a - i))
        i_a_new = i_a + (beta * (i - i_a))
        return i_new, i_a_new
    else:
        return i, i_a


def update_opinions_before(i, i_b, threshold, beta):
    '''
       This function controls the neighbour before the selected person and updates
       that persons opinion using the equation given if their opinion is in the threshold
       '''
    if abs(i - i_b) < threshold:
        i_new = i + (beta * (i_b - i))
        i_b_new = i_b + (beta * (i - i_b))
        return i_new, i_b_new
    else:
        return i, i_b


def parameters(population, population_size, beta, threshold, iterations):
    '''
    This function calls the population_indexes function to get the random person in the array,
    then it randomly picks a number 1 or 2 to determine if the neighbour
    is before or after the individual and gets the updated value and inputs it
    into the array.
    The entire array then gets stored in population_history variable for graphing later.
    It then calls both plot functions which display the graphs respectively
    '''
    population_history = []
    for _ in range(iterations):
        i_index = random.randint(0, population_size - 1)
        i_b, i, i_a, i_b_index, i_a_index = population_indexes(population, population_size, i_index)

        rand_counter = random.randint(1, 2)
        if rand_counter == 1:
            i_new, i_a_new = update_opinions_after(i, i_a, threshold, beta)
            population[i_index] = i_new
            population[i_a_index] = i_a_new
        elif rand_counter == 2:
            i_new, i_b_new = update_opinions_before(i, i_b, threshold, beta)
            population[i_index] = i_new
            population[i_b_index] = i_b_new

        population_history.append(population.copy())

    plot_histogram(population)
    plot_opinions(population_history)



def initial_population(population_size):
    '''
    This function creates the numpy array of numbers from 0 to 1
    with the given population size
    '''
    population = np.random.rand(population_size)
    return population

def defuant_main(beta, threshold):
    '''
    This function gets each initial condition from the parser and runs the parameters function
    with the initial conditions
    '''



    population_size = 100
    population = initial_population(population_size)
    parameters(population, population_size, beta, threshold, iterations=1000)
    print(threshold, "is the threshold value,", beta, "is the beta value")


def test_defuant():
    '''
    Calls the individual test functions for each individual function
    '''
    test_initial_population()
    test_update_opinions()
    test_population_indexes()

def test_initial_population():
    '''
    Runs the test functions for the initial_population function
    '''
    population_size = 50
    assert (len(initial_population(population_size))) == population_size
    print("Initial Population Test function passed")

def test_update_opinions():
    '''
    Runs the test functions for the update_opinions function
    '''
    assert (update_opinions_after(0.8, 0.6, 0.5, 0.5)) == (0.7, 0.7)
    assert (update_opinions_after(0.8, 0.1, 0.5, 0.5)) == (0.8, 0.1)

    assert (update_opinions_before(0.2, 0.6, 0.6, 0.2)) == (0.28, 0.52)
    assert (update_opinions_before(0.3, 0.9, 0.3, 0.3)) == (0.3, 0.9)

    print("Updated opinions test passed")

def test_population_indexes():
    '''
    Runs the test functions for the population_indexes function
    '''
    assert (population_indexes(population = [0.1, 0.7, 0.3, 0.9, 0.4, 0.5, 0.5, 0.2, 0.4, 0.8],
    population_size = 10, i_index = 2) == (0.7, 0.3, 0.9, 1, 3))
    assert (population_indexes(population=[0.1, 0.7, 0.3, 0.9, 0.4, 0.5, 0.5, 0.2, 0.4, 0.8],
    population_size=10, i_index=0) == (0.8, 0.1, 0.7, 9, 1))
    print("Population Indexes test passed")

'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''


def main():

    parser = argparse.ArgumentParser()

    # Task 1 command line parameters
    parser.add_argument("-ising_model", action='store_true', help="Ising model with default parameters")
    parser.add_argument("-external", type=float, default=0.0,
                        help="Ising external value. Defaults to 0")
    parser.add_argument("-alpha", type=float, default=1,
                        help="Ising temperature value. Defaults to 1")
    parser.add_argument("-test_ising", action='store_true', help="Run Ising tests")


    # Task 2 command line parameters
    parser.add_argument("-defuant", action='store_true', help="Defuant model with default parameters")
    parser.add_argument("-beta", type=float, default=0.5,
                        help="Defuant beta value. Defaults to 0.5")
    parser.add_argument("-threshold", type=float, default=0.5,
                        help="Defuant threshold value. Defaults to 0.5")
    parser.add_argument("-test_defuant", action='store_true', help="Run defuant tests")


    # Task 3 command line parameters
    parser.add_argument("-network", type=int, help="Create a random network, size of n")
    parser.add_argument("-test_network", action='store_true', help="Run network tests")


    # Task 4 command line parameters
    parser.add_argument("-random_network", type=int, help="Create a random network, size of n")
    parser.add_argument("-connection_probability", type=float, default=0.3,
                        help="Connection probability. Defaults to 0.3")
    parser.add_argument("-ring_network", type=int, help="Create a ring network with a range of 1 and a size of n")
    parser.add_argument("-range", type=int, default=2, help="Network range. Defaults to 2")
    parser.add_argument("-small_world", type=int, help="Create a small-worlds network with default parameters, size n")
    parser.add_argument("-re_wire", type=float, default=0.2, help="Re-wire probability. Defaults to 0.2")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return

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
        network.make_random_network(args.network, args.connection_probability)
        print("Mean degree: " + str(network.get_mean_degree()))
        # print("Average path length: " + str(network.get_mean_path_length()))
        # print("Clustering co - efficient:" + str(network.get_mean_clustering()))
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
