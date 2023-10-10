#!/usr/bin/env python3

from timeit import default_timer as timer

import numpy as np

from crossover import Crossover
from FISAGALS import FISAGALS
from fitness_scaling import FitnessScaling
from mutation import Mutation
from vrp_instance import VRPInstance


def read_cordeau_instance(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the first line to get instance information
    header = lines[0].split()
    n_vehicles, n_customers, n_depots = map(int, header[1:4])
    max_capacity = lines[1].split()[1]
    vrp_instance = VRPInstance(n_vehicles, n_customers, n_depots, max_capacity)
    
    # Initialize NumPy arrays to store customer and depot data
    customer_dtype = np.dtype([
        ('id', int),
        ('x', float),
        ('y', float),
        ('demand', int),
    ])
    customers = np.zeros((n_customers,), dtype=customer_dtype)

    depot_dtype = np.dtype([
        ('id', int),
        ('x', float),
        ('y', float),
    ])
    depots = np.zeros((n_depots,), dtype=depot_dtype)

    # Read customer data
    for i, line in enumerate(lines[n_vehicles+1:n_vehicles+1 + n_customers]):
        data = line.split()
        if len(data) >= 5:
            customers[i]['id'] = int(data[0])
            customers[i]['x'] = float(data[1])
            customers[i]['y'] = float(data[2])
            customers[i]['demand'] = int(data[4])

    # Read depot data
    for i, line in enumerate(lines[n_vehicles+1 + n_customers:]):
        data = line.split()
        if len(data) >= 3:
            depots[i]['id'] = int(data[0])
            depots[i]['x'] = float(data[1])
            depots[i]['y'] = float(data[2])
  
    return vrp_instance, customers, depots

if __name__ == "__main__":
    # mutation_rate = 1.0
    # crossover_rate = 1.0
    # vrp_instance = VRPInstance(4, 12, 2, 80)
    # mutation = Mutation(vrp_instance, mutation_rate)
    # crossover = Crossover(vrp_instance, crossover_rate)
    # for i in range(100):
    #     chromosome1 = np.array([3, 1, 2, 4, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])      
    #     chromosome2 = np.array([1, 3, 3, 3, 4, 2, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])      

    #     start = timer()
    #     # mutation.uniform(chromosome1)
    #     crossover_new1 = crossover.order(chromosome1, chromosome2)
    #     end = timer()
        
    #     # print(end - start)
    #     # print(f"{chromosome1}")
    #     print(crossover_new1)

    # Define parameters
    instance_file_path = "../benchmark/C-mdvrp/p01"        
    # vrp_instance, customers, depots = read_cordeau_instance(instance_file_path)
    vrp_instance = VRPInstance(4, 12, 2, 80)
    population_size = 100
    crossover_rate = 0.8
    mutation_rate = 0.1
    max_generations = 1000
    fitness_scaling = FitnessScaling.power_rank

    # Run it GA
    ga = FISAGALS(vrp_instance, population_size, crossover_rate, mutation_rate, max_generations, fitness_scaling)
    ga.run()