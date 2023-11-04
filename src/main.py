#!/usr/bin/env python3

import numpy as np

from FISAGALS import FISAGALS
from fitness_scaling import power_rank
from selection import n_tournaments
from local_search import two_opt
from vrp import Customer, Depot, VRPInstance


def read_cordeau_instance(file_path: str) -> VRPInstance:
    """
    Reads benchmark data from cordeau
    param: file_path - location to benchmark data
    return: vrp instance
    """

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the first line to get instance information
    header = lines[0].split()
    n_vehicles, n_customers, n_depots = map(int, header[1:4])
    max_capacity = int(lines[1].split()[1])

    customers = np.zeros((n_customers,), dtype=Customer)
    depots = np.zeros((n_depots,), dtype=Depot)

    # Read customer data
    for i, line in enumerate(lines[n_depots + 1: n_depots + 1 + n_customers]):
        data = line.split()
        if len(data) >= 5:
            customer = Customer(int(data[0]), float(data[1]), float(data[2]), int(data[4]))
            customers[i] = customer

    # Read depot data
    for i, line in enumerate(lines[n_depots + 1 + n_customers:]):
        data = line.split()
        if len(data) >= 3:
            # depot id is + n_customers offset, unfavorable in later stages of GA
            depot = Depot(int(data[0]) - n_customers, float(data[1]), float(data[2]))
            depots[i] = depot

    return VRPInstance(n_vehicles, n_customers, n_depots, max_capacity, customers, depots)


# def test_operators():
#     """
#     Utility function performing operations on simple and small vrp instance for easier evaluating weather correct
#     """
#
#     n_customers = 12
#     n_depots = 2
#     customers = np.zeros((n_customers,), dtype=Customer)
#     depots = np.zeros((n_depots,), dtype=Depot)
#     for i in range(12):
#         customer = Customer(i + 1, i * 10, i * 5, (1 + i) * 5)
#         customers[i] = customer
#     for i in range(2):
#         depot = Depot(i + 1, 10 ** (i + 1), 50 * i)
#         depots[i] = depot
#     vrp_instance = VRPInstance(4, n_customers, n_depots, 80, customers, depots)
#
#     mutation_rate = 1.0
#     crossover_rate = 1.0
#     mutation = Mutation(vrp_instance, mutation_rate)
#     crossover = Crossover(vrp_instance, crossover_rate)
#     for i in range(100):
#         chromosome1 = np.array([3, 1, 2, 4, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
#         chromosome2 = np.array([1, 3, 3, 3, 4, 2, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
#
#         mutation.inversion(chromosome1)
#         # crossover_new1 = crossover.uniform(chromosome1, chromosome2)


if __name__ == "__main__":
    # Set the print options to control the display format
    np.set_printoptions(threshold=np.inf)

    # Create vrp instance
    INSTANCE_FILE_PATH = "../benchmark/C-mdvrp/p04"
    VRP_INSTANCE = read_cordeau_instance(INSTANCE_FILE_PATH)

    # Set GA parameters
    POPULATION_SIZE = 100
    CROSSOVER_RATE = 0.5
    MUTATION_RATE = 0.5
    MAX_GENERATIONS = 200
    FITNESS_SCALING = power_rank
    SELECTION_METHOD = n_tournaments
    LOCAL_SEARCH = two_opt
    tournament_size = 1
    elitism_percentage = 0.05

    # Configure GA and run
    ga = FISAGALS(VRP_INSTANCE,
                  POPULATION_SIZE,
                  CROSSOVER_RATE,
                  MUTATION_RATE,
                  MAX_GENERATIONS,
                  FITNESS_SCALING,
                  SELECTION_METHOD,
                  LOCAL_SEARCH,
                  tournament_size=tournament_size,
                  elitism_percentage=elitism_percentage)
    ga.run()
