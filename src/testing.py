import numpy as np

from src.crossover import Crossover
from src.mutation import Mutation
from src.vrp import Customer, Depot, VRPInstance

n_customers = 40
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
mutation = Mutation(vrp_instance, mutation_rate)
crossover = Crossover(vrp_instance, crossover_rate)
crossover.adaptive_crossover_rate = 1.0
mutation.adaptive_mutation_rate = 1.0
for i in range(100):
    random_permutation = np.random.permutation(np.arange(1, n_customers + 1))
    chromosome1 = np.concatenate((np.array([9, 7, 13, 11]), random_permutation))
    chromosome2 = np.concatenate((np.array([12, 9, 11, 8]), random_permutation))

    # mutation.inversion(chromosome1)
    crossover_new1 = crossover.periodic_crossover_with_insertions(chromosome1, chromosome2)
    print(f"c1: {crossover_new1}")
