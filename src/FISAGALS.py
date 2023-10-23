import random
from typing import Callable

import numpy as np
from numpy import ndarray

from crossover import Crossover
from mutation import Mutation
from plot import plot_fitness, plot_routes
from vrp import Customer, Depot, VRPInstance
import datetime


class FISAGALS:
    """
    - Fitness-scaling adaptive genetic algorithm with local search
    - Chromosome representation specific integer string consisting of three parts:
        1. Number of vehicles for each depot
        1. Number of customers for each vehicle to serve
        2. The order of customers for each vehicle to serve
        E.g. for 2 depots with 3 vehicles and 7 customers (2, 1, 2, 3, 2, 1, 2, 3, 4, 5, 6, 7)
        => first depot (index 0) has 2 vehicles, first vehicle (index 2) serves customer 1 and 2 (index 5 and 6)
        => second depot (Index 1) has 1 vehicles (index 2), serving customer 6 and 7 (index 10, 11)
    """

    THRESHOLD = 200
    TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def __init__(self, vrp_instance: VRPInstance,
                 population_size: int,
                 crossover_rate: float,
                 mutation_rate: float,
                 max_generations: int,
                 fitness_scaling: Callable[[ndarray], ndarray],
                 selection_method: Callable[[ndarray, ndarray, int], ndarray],
                 tournament_size: int = 5,
                 elitism_percentage: float = 0.1,
                 k1: float = 1.0,
                 k2: float = 0.5):
        """
            param: k1 and k2 - control rates for adaptive genetic operators
        """
        self.vrp_instance: VRPInstance = vrp_instance
        self.population_size = population_size
        self.crossover = Crossover(self.vrp_instance, crossover_rate)
        self.mutation = Mutation(self.vrp_instance, mutation_rate)
        self.max_generations = max_generations
        self.fitness_scaling: Callable[[ndarray], ndarray] = fitness_scaling
        self.selection_method: Callable[[ndarray, ndarray, int], ndarray] = selection_method
        self.fitness_stats = np.zeros(max_generations, dtype=np.dtype([("max", float), ("avg", float), ("min", float)]))
        self.tournament_size = tournament_size
        self.elitism_percentage = elitism_percentage
        self.k1 = k1
        self.k2 = k2

    def generate_initial_population(self) -> ndarray:
        """
        Random initial population
        return: 2D array with all chromosome in the population
        """

        initial_population = []
        for _ in range(self.population_size):
            # Part 1: Number of vehicles for each depot
            depot_vehicle_count = np.zeros(self.vrp_instance.n_depots, dtype=int)
            for i in range(self.vrp_instance.n_vehicles):
                # depot_index = np.random.randint(self.vrp_instance.n_depots)
                # depot_vehicle_count[depot_index] += 1
                depot_vehicle_count[i] = 1

            # Part 2: Number of customers for each vehicle
            vehicle_customer_count = np.zeros(self.vrp_instance.n_vehicles, dtype=int)
            total_customers_assigned = 0
            avg_customers_per_vehicle = self.vrp_instance.n_customers / self.vrp_instance.n_vehicles
            std_deviation = 1.0

            # One additional loop to guarantee all customers are assigned to vehicles
            for i in range(self.vrp_instance.n_vehicles + 1):
                # Calculate the maximum number of customers that can be assigned to this vehicle
                max_customers = self.vrp_instance.n_customers - total_customers_assigned
                if max_customers < 1:
                    break

                # Excluding the additional loop
                if i < self.vrp_instance.n_vehicles:
                    # Generate a random number of customers for this vehicle using a Gaussian distribution
                    # centered around the avg_customers_per_vehicle
                    num_customers = int(np.random.normal(loc=avg_customers_per_vehicle, scale=std_deviation))
                    # Ensure it's within valid bounds
                    num_customers = max(1, min(max_customers, num_customers))
                    vehicle_customer_count[i] = num_customers
                else:
                    # If all vehicles assigned and customers remain, assign the rest to random vehicle
                    num_customers = max_customers
                    i = np.random.randint(self.vrp_instance.n_vehicles)
                    vehicle_customer_count[i] += num_customers

                total_customers_assigned += num_customers

            # Part 3: Random order of customers for each vehicle
            order_of_customers = np.random.permutation(np.arange(1, self.vrp_instance.n_customers + 1))

            # Combine the three parts to form a chromosome
            chromosome = np.concatenate((depot_vehicle_count, vehicle_customer_count, order_of_customers))
            initial_population.append(chromosome)

        return np.array(initial_population, dtype=int)

    def evaluate_fitness(self, chromosome: ndarray) -> float:
        """
        Fitness evaluation of a single chromosome
        param: chromosome 1D array
        return: fitness value
        """

        # fitness: total distance
        fitness = 0.0
        depot_index = 0
        vehicle_index = self.vrp_instance.n_depots
        customer_index = self.vrp_instance.n_depots + self.vrp_instance.n_vehicles
        # keep track of iterations of a depot
        depot_value_counter = 1

        for i in range(self.vrp_instance.n_vehicles):
            vehicle_i_n_customers = chromosome[vehicle_index + i]
            # Capacity for every vehicle the same at the moment. TODO dynamic capacity which vehicle class
            vehicle_i_capacity = 0

            # Check if all iterations for vehicles of current depot are done. Then continue with next depot
            if depot_value_counter > chromosome[depot_index]:
                depot_value_counter = 1
                depot_index += 1

            vehicle_i_depot: Depot = self.vrp_instance.depots[depot_index]

            for j in range(vehicle_i_n_customers):
                customer_value1 = chromosome[customer_index + j]
                # Indexing of customers starts with 1 not 0, so -1 necessary
                customer_1: Customer = self.vrp_instance.customers[customer_value1 - 1]

                # First iteration in loop: first trip 
                if j == 0:
                    # Add distance from depot to customer with the euclidean distance.
                    # Assuming single customer demand <= vehicle max capacity
                    fitness += np.linalg.norm(
                        np.array([vehicle_i_depot.x, vehicle_i_depot.y]) - np.array([customer_1.x, customer_1.y]))

                    # TODO add capacity constraint meaning vehicles with different capacity
                    # Thus customer demand > vehicle max capacity possible but at least 1 vehicle exists with greater capacity
                    vehicle_i_capacity += customer_1.demand

                # Check if next customer exists in route exists
                if j < vehicle_i_n_customers - 1:
                    customer_value2 = chromosome[customer_index + j + 1]
                    customer_2: Customer = self.vrp_instance.customers[customer_value2 - 1]

                    # Check customer_2 demand exceeds vehicle capacity limit
                    # TODO Add heterogeneous capacity for vehicles
                    if vehicle_i_capacity + customer_2.demand > self.vrp_instance.max_capacity:
                        # Trip back to depot necessary. Assuming heading back to same depot it came from
                        # TODO visit different depot if possible e.g. AF-VRP charging points for robots
                        fitness += np.linalg.norm(
                            np.array([customer_1.x, customer_1.y]) - np.array([vehicle_i_depot.x, vehicle_i_depot.y]))

                        # from depot to next customer
                        fitness += np.linalg.norm(
                            np.array([vehicle_i_depot.x, vehicle_i_depot.y]) - np.array([customer_2.x, customer_2.y]))
                        vehicle_i_capacity = 0
                    else:
                        # Add distance between customers
                        fitness += np.linalg.norm(
                            np.array([customer_1.x, customer_1.y]) - np.array([customer_2.x, customer_2.y]))

                    vehicle_i_capacity += customer_2.demand

                # Last iteration in loop, add trip from last customer to depot
                if j >= vehicle_i_n_customers - 1:
                    fitness += np.linalg.norm(
                        np.array([customer_1.x, customer_1.y]) - np.array([vehicle_i_depot.x, vehicle_i_depot.y]))

            customer_index += vehicle_i_n_customers
            depot_value_counter += 1

        return fitness

    def log_configuration(self, generation, chromosome):
        with open(f'../results/{self.__class__.__name__}/{self.TIMESTAMP}/best_chromosome.txt', 'a') as file:
            file.write(f'Population size: {self.population_size}'
                       f'\nGenerations: {self.max_generations}'
                       f'\nMutation and crossover: adaptive'
                       f'\nFitness scaling: {self.fitness_scaling.__name__}'
                       f'\nSelection method: {self.selection_method.__name__}'
                       f'\nTournament size: {self.tournament_size}'
                       f'\nElitism: {self.elitism_percentage}'
                       f'\nFitness: {self.fitness_stats["min"][generation]:.2f}'
                       f'\nBest chromosome: ')
            np.savetxt(file, chromosome, fmt='%d', newline=' ')

    def run(self):
        """
        Execution of FISAGALS
        """

        population = self.generate_initial_population()

        for generation in range(self.max_generations):
            # Fitness evaluation and scaling
            fitness_scores = np.array(
                [(int(i), self.evaluate_fitness(chromosome)) for i, chromosome in enumerate(population)],
                dtype=np.dtype([("index", int), ("fitness", float)]))

            # Save statistics about raw fitness
            self.fitness_stats[generation]["max"] = np.max(fitness_scores['fitness'])
            self.fitness_stats[generation]["avg"] = np.mean(fitness_scores['fitness'])
            self.fitness_stats[generation]["min"] = np.min(fitness_scores['fitness'])

            self.fitness_scaling(fitness_scores)

            # Parent selection
            # before starting the parent selection. Save percentage of best individuals
            # TODO numpy arraay top_chromosome
            top_individuals_i = fitness_scores[:int(self.population_size * self.elitism_percentage)]
            top_chromosome_i = []
            for i, index in enumerate(top_individuals_i):
                top_chromosome_i.append(population[index[0]])

            self.selection_method(population, fitness_scores, self.tournament_size)

            # Elitism: Replace the some percentage of worst individuals with the best individuals
            worst_individuals_i = np.argpartition(fitness_scores["fitness"],
                                                  int(self.population_size * self.elitism_percentage))[
                                  :int(self.population_size * self.elitism_percentage)]
            for i, worst_i in enumerate(worst_individuals_i):
                population[worst_i] = top_chromosome_i[i]
                fitness_scores["fitness"][worst_i] = top_individuals_i["fitness"][i]

            # Crossover
            children = np.empty((self.population_size,
                                 self.vrp_instance.n_depots + self.vrp_instance.n_vehicles + self.vrp_instance.n_customers),
                                dtype=population.dtype)
            for i in range(0, self.population_size, 2):
                # Adaptive rates for genetic operators
                min_parent_fitness = min(fitness_scores["fitness"][i], fitness_scores["fitness"][i + 1])
                if min_parent_fitness <= self.fitness_stats[generation]["avg"]:
                    max_parent_fitness = max(fitness_scores["fitness"][i], fitness_scores["fitness"][i + 1])
                    numerator = min_parent_fitness - self.fitness_stats[generation]["min"]
                    denominator = max_parent_fitness - self.fitness_stats[generation]["min"]

                    self.crossover.adaptive_crossover_rate = self.k1 * (numerator / denominator)
                    self.mutation.adaptive_mutation_rate = self.k2 * (numerator / denominator)
                else:
                    self.crossover.adaptive_crossover_rate = self.k1
                    self.mutation.adaptive_mutation_rate = self.k2

                # Generate children, second child by swapping parents
                children[i] = self.crossover.order(self.crossover.uniform(population[i], population[i + 1]),
                                                   population[i + 1])
                children[i + 1] = self.crossover.order(self.crossover.uniform(population[i + 1], population[i]),
                                                       population[i])

            # Mutation
            for i in range(0, self.population_size):
                self.mutation.uniform(children[i])
                rand_num = random.random()
                if rand_num < 0.33:
                    self.mutation.swap(children[i])
                elif 0.33 <= rand_num < 0.66:
                    self.mutation.inversion(children[i])
                else:
                    self.mutation.insertion(children[i])

            # TODO add local search

            # Replace old generation with new generation
            population = np.copy(children)

            # Termination convergence criteria
            print(f"{generation}")
            # fitness_bound = self.fitness_stats["min"][generation - int(self.max_generations*0.3)] if generation - int(self.max_generations*0.3) >= 0 else float('inf')
            # if self.fitness_stats["min"][generation] - self.THRESHOLD > fitness_bound:
            #     break

        plot_fitness(self)
        # get and plot best individual
        min_index = np.argmin(fitness_scores["fitness"])
        plot_routes(self, population[fitness_scores[min_index]["index"]])
        self.log_configuration(generation, population[fitness_scores[min_index]["index"]])

        # Return the best solution found
        print(population[0])
