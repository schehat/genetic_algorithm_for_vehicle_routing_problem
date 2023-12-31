import timeit

import numpy as np

from src.GA import GA
from src.crossover import Crossover
from src.fitness_scaling import power_rank
from src.initial_population import initial_population_grouping_savings_nnh
from src.local_search import two_opt
from src.main import read_cordeau_instance
from src.mutation import Mutation
from src.selection import n_tournaments
from src.vrp import Customer, Depot, VRPInstance

n_customers = 48
n_depots = 4
customers = np.zeros((n_customers,), dtype=Customer)
depots = np.zeros((n_depots,), dtype=Depot)
for i in range(n_customers):
    customer = Customer(i + 1, i * 10, i * 5, 10, (1 + i) * 5, 0, 1000)
    customers[i] = customer
for i in range(n_depots):
    depot = Depot(i + 1, 10 ** (i + 1), 50 * i, 0, 1000)
    depots[i] = depot
vrp_instance = VRPInstance(4, n_customers, n_depots, 80, customers, depots, 500)

mutation_rate = 1.0
crossover_rate = 1.0
mutation = Mutation(vrp_instance)
crossover = Crossover(vrp_instance)
crossover.adaptive_crossover_rate = 1.0
mutation.adaptive_mutation_rate = 1.0

# Create vrp instance
INSTANCE_FILE_PATH = "../benchmark/c-mdvrptw/pr01"
# VRP_INSTANCE = VRPInstance(4, n_customers, n_depots, 200, customers, depots, 500)
VRP_INSTANCE = read_cordeau_instance(INSTANCE_FILE_PATH)

# Set GA parameters
POPULATION_SIZE = 40
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.5
MAX_GENERATIONS = 1
INITIAL_POPULATION = initial_population_grouping_savings_nnh
# INITIAL_POPULATION = initial_population_random
FITNESS_SCALING = power_rank
SELECTION_METHOD = n_tournaments
LOCAL_SEARCH_COMPLETE = two_opt
tournament_size = 2
elitism_percentage = 0.1

# Configure GA and run
ga = GA(VRP_INSTANCE,
        POPULATION_SIZE,
        MAX_GENERATIONS,
        INITIAL_POPULATION,
        FITNESS_SCALING,
        SELECTION_METHOD,
        LOCAL_SEARCH_COMPLETE,
        tournament_size=tournament_size,
        p_elitism=elitism_percentage)

# for i in range(1):
#     random_permutation = np.random.permutation(np.arange(1, n_customers + 1))
#     chromosome1 = np.concatenate((np.array([9, 7, 13, 11]), random_permutation))
#     chromosome2 = np.concatenate((np.array([12, 9, 11, 8]), random_permutation))
#
#
#     # mutation.inversion(chromosome1)
#
#     ga.education.chromosome = chromosome1
#     print(f"c1: {chromosome1}")
#     print(f"c1: {ga.education.run(chromosome1)}")

random_permutation = np.random.permutation(np.arange(1, n_customers + 1))
chromosome1 = np.concatenate((np.array([10, 7, 13, 18]), random_permutation))
chromosome2 = np.concatenate((np.array([12, 15, 13, 8]), random_permutation))
ga.education.chromosome = chromosome1


def ga_run():
    ga.education.run(chromosome1)


execution_time = timeit.timeit(ga_run, number=10)
print(f"Execution time: {execution_time} seconds")
