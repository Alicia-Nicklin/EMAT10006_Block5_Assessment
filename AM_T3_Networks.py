import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
        Calculates the average of the degree of all nodes in the network
        '''

        count = 0 # initializes variable to sum number of nodes

        for node in self.nodes: # iterates through nodes in network
                count += sum(node.connections) # adds number of connections to count variable

        # calculates average by dividing count by number of nodes
        return count / len(self.nodes)


    def get_clustering(self):
        clustering = []
        true_connections = -1
        for node in self.nodes:
            neighbours = node.connections
            n = sum(neighbours) #calculates the number of neighbours
            possible_connections = n * (n-1) / 2

            if n <= 1:
                clustering.append(0) #case where there are one or no neighbours so no triangles can be formed
                continue

            for i in range(n):
                for j in range(n):
                    neighbor_i = self.nodes[neighbours[i]]
                    neighbor_j = self.nodes[neighbours[j]]
                    for common_neighbor in neighbours:
                        if common_neighbor in neighbor_i.connections and common_neighbor in neighbor_j.connections:
                            true_connections += 1
            clustering.append(true_connections / possible_connections)

        return clustering



    def get_mean_clustering(self):
        mean_clustering = sum(self.get_clustering()) / (len(self.get_clustering()))
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

    # def make_ring_network(self, N, neighbour_range=1):
    #
    # # Your code  for task 4 goes here
    #
    # def make_small_world_network(self, N, re_wire_prob=0.2):
    #
    # # Your code for task 4 goes here

    def plot(self):

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

            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')


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
    for node in network.nodes:
        print(node.connections)
    network.plot()
    plt.show()
    print("Testing ring network")
    assert (network.get_mean_degree() == 2), network.get_mean_degree()
    assert (network.get_mean_clustering() == 0), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 2.777777777777778), network.get_mean_path_length()

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
    assert (network.get_mean_clustering() == 0), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 5), network.get_mean_path_length()

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
    assert (network.get_mean_clustering() == 1), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 1), network.get_mean_path_length()

    print("All tests passed")


# '''
# ==============================================================================================================
# This section contains code for the Ising Model - task 1 in the assignment
# ==============================================================================================================
# '''
#
#
# def calculate_agreement(population, row, col, external=0.0):
#     '''
#     This function should return the *change* in agreement that would result if the cell at (row, col) was to flip it's value
#     Inputs: population (numpy array)
#             row (int)
#             col (int)
#             external (float)
#     Returns:
#             change_in_agreement (float)
#     '''
#
#     # Your code for task 1 goes here
#
#     return np.random * population
#
#
# def ising_step(population, external=0.0):
#     '''
#     This function will perform a single update of the Ising model
#     Inputs: population (numpy array)
#             external (float) - optional - the magnitude of any external "pull" on opinion
#     '''
#
#     n_rows, n_cols = population.shape
#     row = np.random.randint(0, n_rows)
#     col = np.random.randint(0, n_cols)
#
#     agreement = calculate_agreement(population, row, col, external=0.0)
#
#     if agreement < 0:
#         population[row, col] *= -1
#
#
# # Your code for task 1 goes here
#
# def plot_ising(im, population):
#     '''
#     This function will display a plot of the Ising model
#     '''
#
#
# new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
# im.set_data(new_im)
# plt.pause(0.1)
#
#
# def test_ising():
#     '''
#     This function will test the calculate_agreement function in the Ising model
#     '''
#
#
# print("Testing ising model calculations")
# population = -np.ones((3, 3))
# assert (calculate_agreement(population, 1, 1) == 4), "Test 1"
#
# population[1, 1] = 1.
# assert (calculate_agreement(population, 1, 1) == -4), "Test 2"
#
# population[0, 1] = 1.
# assert (calculate_agreement(population, 1, 1) == -2), "Test 3"
#
# population[1, 0] = 1.
# assert (calculate_agreement(population, 1, 1) == 0), "Test 4"
#
# population[2, 1] = 1.
# assert (calculate_agreement(population, 1, 1) == 2), "Test 5"
#
# population[1, 2] = 1.
# assert (calculate_agreement(population, 1, 1) == 4), "Test 6"
#
# "Testing external pull"
# population = -np.ones((3, 3))
# assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
# assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
# assert (calculate_agreement(population, 1, 1, 10) == 14), "Test 9"
# assert (calculate_agreement(population, 1, 1, -10) == -6), "Test 10"
#
# print("Tests passed")
#
#
# def ising_main(population, alpha=None, external=0.0):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_axis_off()
#     im = ax.imshow(population, interpolation='none', cmap='RdPu_r')
#
#     # Iterating an update 100 times
#     for frame in range(100):
#         # Iterating single steps 1000 times to form an update
#         for step in range(1000):
#             ising_step(population, external)
#         print('Step:', frame, end='\r')
#         plot_ising(im, population)
#
#
# '''
# ==============================================================================================================
# This section contains code for the Defuant Model - task 2 in the assignment
# ==============================================================================================================
# '''
#
#
# def defuant_main():
#
#
# # Your code for task 2 goes here
#
# def test_defuant():
#
#
# # Your code for task 2 goes here
#
#
# '''
# ==============================================================================================================
# This section contains code for the main function- you should write some code for handling flags here
# ==============================================================================================================
# '''
#
#
# def main():
#
#
# # You should write some code for handling flags here
#
#
# if __name__ == "__main__":
#     main()


test_networks()



