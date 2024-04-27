import argparse
import numpy as np
import matplotlib.pyplot as plt
import random


def parseArgs():
    '''
    This function receives any inputs from the command line
    and parses them to store the inputs
    '''
    parser = argparse.ArgumentParser(description="Runs the defuant model")
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--coupling', type=float)
    parser.add_argument('--test_defuant', type=str, help="Runs the test functions")
    return parser.parse_args()


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

def update_opinions_after(i, i_a, threshold, coupling):
    '''
    This function controls the neighbour after the selected person and updates
    that persons opinion using the equation given if their opinion is in the threshold
    '''
    if abs(i - i_a) < threshold:
        i_new = i + (coupling * (i_a - i))
        i_a_new = i_a + (coupling * (i - i_a))
        return i_new, i_a_new
    else:
        return i, i_a


def update_opinions_before(i, i_b, threshold, coupling):
    '''
       This function controls the neighbour before the selected person and updates
       that persons opinion using the equation given if their opinion is in the threshold
       '''
    if abs(i - i_b) < threshold:
        i_new = i + (coupling * (i_b - i))
        i_b_new = i_b + (coupling * (i - i_b))
        return i_new, i_b_new
    else:
        return i, i_b


def parameters(population, population_size, coupling, threshold, iterations):
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
            i_new, i_a_new = update_opinions_after(i, i_a, threshold, coupling)
            population[i_index] = i_new
            population[i_a_index] = i_a_new
        elif rand_counter == 2:
            i_new, i_b_new = update_opinions_before(i, i_b, threshold, coupling)
            population[i_index] = i_new
            population[i_b_index] = i_b_new

        population_history.append(population.copy())

    plot_histogram(population)
    plot_opinions(population_history)

def get_conditions(args):
    '''
    This functions takes the input of the args from the command line and sees
    if a value is inputted else it returns the default preset numbers.
    '''

    coupling = args.coupling if args.coupling else 0.5
    threshold = args.threshold if args.threshold else 0.5


    return coupling, threshold


def initial_population(population_size):
    '''
    This function creates the numpy array of numbers from 0 to 1
    with the given population size
    '''
    population = np.random.rand(population_size)
    return population






def main():
    '''
    This function gets each initial condition from the parser and runs the parameters function
    with the initial conditions
    '''
    args = parseArgs()
    coupling, threshold = get_conditions(args)

    population_size = 100
    population = initial_population(population_size)
    parameters(population, population_size, coupling, threshold, iterations=1000)
    print(threshold, "is the threshold value,", coupling, "is the coupling value")
    test_defuant()

def test_defuant():
    '''
    Calls the individual test functions for each individual function
    '''
    test_conditions()
    test_initial_population()
    test_update_opinions()
    test_population_indexes()
def test_conditions():
    '''
    This code does runs the test function for the conditions to test if they take inputs and if they take no inputs
    '''
    assert (get_conditions(argparse.Namespace(coupling=None, threshold=None))) == (0.5, 0.5)
    assert (get_conditions(argparse.Namespace(coupling=0.8, threshold=None))) == (0.8, 0.5)
    assert (get_conditions(argparse.Namespace(coupling=0.8, threshold=0.1))) == (0.8, 0.1)

    print("Conditions Test function passed")


def test_initial_population():
    population_size = 50
    assert (len(initial_population(population_size))) == population_size
    print("Initial Population Test function passed")

def test_update_opinions():
    assert (update_opinions_after(0.8, 0.6, 0.5, 0.5)) == (0.7, 0.7)
    assert (update_opinions_after(0.8, 0.1, 0.5, 0.5)) == (0.8, 0.1)


    assert (update_opinions_before(0.2, 0.6, 0.6, 0.2)) == (0.28, 0.52)
    assert (update_opinions_before(0.3, 0.9, 0.3, 0.3)) == (0.3, 0.9)

    print("Updated opinions test passed")

def test_population_indexes():

    assert (population_indexes(population = [0.1, 0.7, 0.3, 0.9, 0.4, 0.5, 0.5, 0.2, 0.4, 0.8],
    population_size = 10, i_index = 2) == (0.7, 0.3, 0.9, 1, 3))
    assert (population_indexes(population=[0.1, 0.7, 0.3, 0.9, 0.4, 0.5, 0.5, 0.2, 0.4, 0.8],
    population_size=10, i_index=0) == (0.8, 0.1, 0.7, 9, 1))
    print("Population Indexes test passed")


if __name__ == '__main__':
    main()