#!/usr/bin/env python3


import numpy as np

from customer import Customer
from depot import Depot
from FISAGALS import FISAGALS
from fitness_scaling import FitnessScaling
from src.selection import Selection
from vrp_instance import VRPInstance
from mutation import Mutation
from crossover import Crossover


def read_cordeau_instance(file_path) -> VRPInstance:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the first line to get instance information
    header = lines[0].split()
    n_vehicles, n_customers, n_depots = map(int, header[1:4])
    max_capacity = int(lines[1].split()[1])

    customers = np.zeros((n_customers,), dtype=Customer)
    depots = np.zeros((n_depots,), dtype=Depot)

    # Read customer data
    for i, line in enumerate(lines[n_vehicles + 1: n_vehicles + 1 + n_customers]):
        data = line.split()
        if len(data) >= 5:
            customer = Customer(int(data[0]), float(data[1]), float(data[2]), int(data[4]))
            customers[i] = customer

    # Read depot data
    for i, line in enumerate(lines[n_vehicles + 1 + n_customers:]):
        data = line.split()
        if len(data) >= 3:
            # depot id is + n_customers offset, unfavorable in later stages of GA
            depot = Depot(int(data[0]) - n_customers, float(data[1]), float(data[2]))
            depots[i] = depot

    return VRPInstance(n_vehicles, n_customers, n_depots, max_capacity, customers, depots)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    # Set the print options to control the display format

    # TESTING WITHOUT BENCHMARK
    # n_customers = 12
    # n_depots = 2
    # customers = np.zeros((n_customers,), dtype=Customer)
    # depots = np.zeros((n_depots,), dtype=Depot)
    # for i in range(12):
    #     customer = Customer(i+1, i*10, i*5, (1+i)*5)
    #     customers[i] = customer
    # for i in range(2):
    #     depot = Depot(i+1, 10**(i+1), 50*i)
    #     depots[i] = depot
    # vrp_instance = VRPInstance(4, n_customers, n_depots, 80, customers, depots)

    # TEST MUTATION AND CROSSOVER
    # mutation_rate = 1.0
    # crossover_rate = 1.0
    # mutation = Mutation(vrp_instance, mutation_rate)
    # crossover = Crossover(vrp_instance, crossover_rate)
    # for i in range(100):
    #     chromosome1 = np.array([3, 1, 2, 4, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    #     chromosome2 = np.array([1, 3, 3, 3, 4, 2, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    #
    #     mutation.inversion(chromosome1)
    #     # crossover_new1 = crossover.uniform(chromosome1, chromosome2)
    #
    #     # print(end - start)
    #     print(f"{chromosome1}")
    #     # print(crossover_new1)

    # print("Customer Data:")
    # for customer in vrp_instance.customers:
    #     print(f"Customer {customer.id}")
    #     print(f"  X-coordinate: {customer.x}")
    #     print(f"  Y-coordinate: {customer.y}")
    #     print(f"  Demand: {customer.demand}")

    # # Print depot data
    # print("Depot Data:")
    # for depot in vrp_instance.depots:
    #     print(f"Depot {depot.id}")
    #     print(f"  X-coordinate: {depot.x}")
    #     print(f"  Y-coordinate: {depot.y}")

    instance_file_path = "../benchmark/C-mdvrp/p01"
    vrp_instance = read_cordeau_instance(instance_file_path)

    population_size = 300
    crossover_rate = 0.5
    mutation_rate = 0.5
    max_generations = 2000
    fitness_scaling = FitnessScaling.power_rank
    selection_method = Selection.n_tournament

    # Run it GA
    ga = FISAGALS(vrp_instance,
                  population_size,
                  crossover_rate,
                  mutation_rate,
                  max_generations,
                  fitness_scaling,
                  selection_method)
    ga.run()
