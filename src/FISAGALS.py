import random
from typing import Callable

import numpy as np
from numpy import ndarray

from crossover import Crossover
from mutation import Mutation
from plot import plot_fitness, plot_routes
from purpose import Purpose
from vrp import Customer, Depot, VRPInstance
import datetime


class FISAGALS:
    """
    - Fitness-scaling adaptive genetic algorithm with local search
    - Chromosome representation specific integer string consisting of three parts:
        1. Number of vehicles for each depot
        1. Number of customers for each vehicle to serve
        3. The order of customers for each vehicle to serve
        E.g. for 2 depots with 3 vehicles and 7 customers (2, 1, 2, 3, 2, 1, 2, 3, 4, 5, 6, 7)
        => first depot (index 0) has 2 vehicles, first vehicle (index 2) serves customer 1 and 2 (index 5 and 6)
        => second depot (Index 1) has 1 vehicles (index 2), serving customer 6 and 7 (index 10, 11)
    """

    THRESHOLD = 200
    TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    generation = 0

    def __init__(self, vrp_instance: VRPInstance,
                 population_size: int,
                 crossover_rate: float,
                 mutation_rate: float,
                 max_generations: int,
                 fitness_scaling: Callable[[ndarray], ndarray],
                 selection_method: Callable[[ndarray, ndarray, int], ndarray],
                 local_search,
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
        self.local_search = local_search
        self.tournament_size = tournament_size
        self.elitism_percentage = elitism_percentage
        self.k1 = k1
        self.k2 = k2

        population_type = np.dtype([
            ("individual", int),
            ("chromosome", int,
             (self.vrp_instance.n_depots + self.vrp_instance.n_vehicles + self.vrp_instance.n_customers,)),
            ("fitness", float),
        ])
        self.population = np.zeros(self.population_size, dtype=population_type)
        self.best_solution = np.zeros(1, dtype=population_type)
        self.best_solution[0]["fitness"] = float("inf")
        # Note after scaling order of fitness reversed: min fitness mapped to max_scaled
        self.fitness_stats = np.zeros(max_generations, dtype=np.dtype([("max", float), ("avg", float), ("min", float),
                                                                       ("max_scaled", float), ("avg_scaled", float),
                                                                       ("min_scaled", float)]))

    def run(self):
        """
        Execution of FISAGALS
        """

        self.generate_initial_population()

        for self.generation in range(self.max_generations):
            # Fitness evaluation
            for i, chromosome in enumerate(self.population["chromosome"]):
                self.population[i]["fitness"] = self.evaluate_fitness(chromosome)

            # Save statistics about raw fitness
            self.fitness_stats[self.generation]["max"] = np.max(self.population["fitness"])
            self.fitness_stats[self.generation]["avg"] = np.mean(self.population["fitness"])
            self.fitness_stats[self.generation]["min"] = np.min(self.population["fitness"])

            # Check if there is a new best solution found
            min_fitness = self.fitness_stats[self.generation]["min"]
            if min_fitness < self.best_solution['fitness']:
                self.best_solution = self.population[np.argmin(self.population["fitness"])].copy()

            self.fitness_scaling(self.population)

            # Save statistics about scaled fitness. Note max_scaled corresponds to min raw fitness
            self.fitness_stats[self.generation]["max_scaled"] = np.max(self.population["fitness"])
            self.fitness_stats[self.generation]["avg_scaled"] = np.mean(self.population["fitness"])
            self.fitness_stats[self.generation]["min_scaled"] = np.min(self.population["fitness"])

            # Before starting the parent selection. Save percentage of best individuals
            top_individuals_i = np.argsort(self.population["fitness"])[
                                :int(self.population_size * self.elitism_percentage)]
            top_individuals = self.population[top_individuals_i]

            # Increasing selection pressure over time by increasing tournament size
            if self.generation % (self.max_generations * 0.1) == 0:
                self.tournament_size += 1
            self.selection_method(self.population, self.tournament_size)
            self.do_elitism(top_individuals)

            children = np.empty((self.population_size,
                                 self.vrp_instance.n_depots + self.vrp_instance.n_vehicles + self.vrp_instance.n_customers),
                                dtype=int)

            children = self.do_crossover(children)
            children = self.do_mutation(children)

            # Replace old generation with new generation
            self.population["chromosome"] = children

            # Termination convergence criteria
            print(f"{self.generation}")
            # fitness_bound = self.fitness_stats["min"][generation - int(self.max_generations*0.3)] if generation - int(self.max_generations*0.3) >= 0 else float('inf')
            # if self.fitness_stats["min"][generation] - self.THRESHOLD > fitness_bound:
            #     break

        # get best individual
        min_index = np.argmax(self.population["fitness"])
        best_individual = self.population[min_index]
        # replace scaled fitness to raw fitness
        best_individual["fitness"] = self.evaluate_fitness(best_individual["chromosome"])
        print("individual: before local")
        self.local_search(self, best_individual)
        print("after local")
        print("best: before local")
        self.local_search(self, self.best_solution)
        print("best: after local")

        print(f"min: {np.min(self.fitness_stats['min'])} ?= {self.best_solution}")
        if self.best_solution["fitness"] < np.min(self.fitness_stats["min"]):
            self.fitness_stats[self.max_generations - 1]["min"] = self.best_solution["fitness"]

        plot_fitness(self)
        plot_routes(self, self.best_solution["chromosome"])
        self.log_configuration(self.best_solution)

        # self.population[min_index]["chromosome"] = [1, 1, 1, 1, 14, 19, 8, 9, 44, 45, 33, 15, 37, 17, 42, 19, 40, 41,
        #                                             13, 25, 18, 4,
        #                                             6, 27, 1, 32, 11, 46, 48, 8, 26, 31, 28, 22, 23, 7, 43, 24, 14, 12,
        #                                             47,
        #                                             9, 34, 30, 39, 10, 49, 5, 38,
        #                                             35, 36, 3, 20, 21, 50, 16, 2, 29]
        # plot_routes(self, self.population[min_index]["chromosome"])
        # print(self.evaluate_fitness(self.population[min_index]["chromosome"]))

    def generate_initial_population(self):
        """
        Generate random initial population
        """

        for i in range(self.population_size):
            # Part 1: Number of vehicles for each depot
            depot_vehicle_count = np.zeros(self.vrp_instance.n_depots, dtype=int)
            for depot_index in range(self.vrp_instance.n_depots):
                # For now giving every depot 1 vehicle. TODO make dynamic, more interesting when n_vehicles > n_depots
                # depot_index = np.random.randint(self.vrp_instance.n_depots)
                # depot_vehicle_count[depot_index] += 1
                depot_vehicle_count[depot_index] = 1

            # Part 2: Number of customers for each vehicle
            vehicle_customer_count = np.zeros(self.vrp_instance.n_vehicles, dtype=int)
            total_customers_assigned = 0
            avg_customers_per_vehicle = self.vrp_instance.n_customers / self.vrp_instance.n_vehicles
            std_deviation = 1.0

            # One additional loop to guarantee all customers are assigned to vehicles, relevant in else block
            for vehicle_index in range(self.vrp_instance.n_vehicles + 1):
                # Calculate the maximum number of customers that can be assigned to this vehicle
                max_customers = self.vrp_instance.n_customers - total_customers_assigned
                if max_customers < 1:
                    break

                # Excluding the additional loop
                if vehicle_index < self.vrp_instance.n_vehicles:
                    # Generate a random number of customers for this vehicle using a Gaussian distribution
                    # centered around the avg_customers_per_vehicle
                    num_customers = int(np.random.normal(loc=avg_customers_per_vehicle, scale=std_deviation))
                    # Ensure it's within valid bounds
                    num_customers = max(1, min(max_customers, num_customers))
                    vehicle_customer_count[vehicle_index] = num_customers
                else:
                    # If all vehicles assigned and customers remain, assign the rest to random vehicle
                    num_customers = max_customers
                    vehicle_index = np.random.randint(self.vrp_instance.n_vehicles)
                    vehicle_customer_count[vehicle_index] += num_customers

                total_customers_assigned += num_customers

            # Part 3: Random order of customers for each vehicle
            order_of_customers = np.random.permutation(np.arange(1, self.vrp_instance.n_customers + 1))

            # Combine the three parts to form a chromosome
            chromosome = np.concatenate((depot_vehicle_count, vehicle_customer_count, order_of_customers))
            self.population[i]["individual"] = i
            self.population[i]["chromosome"] = chromosome

    def evaluate_fitness(self, chromosome: ndarray) -> float:
        """
        Calculate fitness for a single chromosome
        param: chromosome - 1D array holding genetic information
        """

        # Fitness is considered the total distance traveled
        total_fitness = 0.0

        def add_fitness(obj1, obj2):
            """
            param: obj1 and obj2 - Customers or Depots
            """
            # use total_fitness declared above
            nonlocal total_fitness
            total_fitness += np.linalg.norm(
                np.array([obj1.x, obj1.y]) - np.array([obj2.x, obj2.y]))

        # While decoding chromosome use add_fitness
        self.decode_chromosome(chromosome, Purpose.FITNESS, add_fitness)
        return total_fitness

    def decode_chromosome(self, chromosome: ndarray, purpose: Purpose, operation: any):
        """
        Decoding chromosome by traversing the genes considering constraints and fetching the routes.
        Expecting a purpose to evaluate which operation should be used
        param: chromosome - 1D array
        param: purpose - defines which operation should be used
        param: operation - function pointer passed to enable maximum flexibility what should happen while decoding.
                           Needs to be in coordination with purpose parameter
        """

        depot_index = 0
        vehicle_index = self.vrp_instance.n_depots
        customer_index = self.vrp_instance.n_depots + self.vrp_instance.n_vehicles
        # keep track of iterations of a depot
        depot_value_counter = 1

        for i in range(self.vrp_instance.n_vehicles):
            vehicle_i_n_customers = chromosome[vehicle_index + i]
            # Capacity for every vehicle the same at the moment. TODO dynamic capacity with vehicle class
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
                    if purpose == purpose.FITNESS:
                        operation(vehicle_i_depot, customer_1)
                    elif purpose == purpose.PLOTTING:
                        operation(vehicle_i_depot, i)
                        operation(customer_1, i)

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
                        if purpose == purpose.FITNESS:
                            operation(customer_1, vehicle_i_depot)
                        elif purpose == purpose.PLOTTING:
                            operation(vehicle_i_depot, i)

                        # from depot to next customer
                        if purpose == purpose.FITNESS:
                            operation(vehicle_i_depot, customer_2)
                        elif purpose == purpose.PLOTTING:
                            operation(customer_2, i)

                        vehicle_i_capacity = 0
                    else:
                        # Add distance between customers
                        if purpose == purpose.FITNESS:
                            operation(customer_1, customer_2)
                        elif purpose == purpose.PLOTTING:
                            operation(customer_2, i)

                    vehicle_i_capacity += customer_2.demand

                # Last iteration in loop, add trip from last customer to depot
                if j >= vehicle_i_n_customers - 1:
                    if purpose == purpose.FITNESS:
                        operation(customer_1, vehicle_i_depot)
                    elif purpose == purpose.PLOTTING:
                        operation(vehicle_i_depot, i)

            customer_index += vehicle_i_n_customers
            depot_value_counter += 1

    def do_elitism(self, top_individuals: ndarray):
        """
        Perform elitism by replacing the worst individuals with the best individuals
        param: top_individuals - structured 3D array ["individual"]["chromosome]["fitness"]
        """

        worst_individuals_i = np.argsort(self.population["fitness"])[
                              :int(self.population_size * self.elitism_percentage)]
        self.population[worst_individuals_i] = top_individuals

    def do_crossover(self, children: ndarray) -> ndarray:
        """
        Handles the crossover
        param: children - empty 1D array
        return: children - 1D array of the generation holding chromosome information
        """

        EXCEPTION_DIVISOR = 4
        self.crossover.adaptive_crossover_rate = self.k1
        self.mutation.adaptive_mutation_rate = self.k2

        for individual in range(0, self.population_size, 2):
            # Adaptive rates for genetic operators
            min_parent_fitness = min(self.population[individual]["fitness"],
                                     self.population[individual + 1]["fitness"])
            if min_parent_fitness >= self.fitness_stats[self.generation]["avg_scaled"]:
                max_parent_fitness = max(self.population[individual]["fitness"],
                                         self.population[individual + 1]["fitness"])
                numerator = min_parent_fitness - self.fitness_stats[self.generation]["min_scaled"]
                denominator = max_parent_fitness - self.fitness_stats[self.generation]["min_scaled"]

                try:
                    self.crossover.adaptive_crossover_rate = self.k1 * (numerator / denominator)
                    self.mutation.adaptive_mutation_rate = self.k2 * (numerator / denominator)
                except ZeroDivisionError:
                    self.crossover.adaptive_crossover_rate = self.k1
                    self.mutation.adaptive_mutation_rate = self.k2
            else:
                self.crossover.adaptive_crossover_rate = self.k1
                self.mutation.adaptive_mutation_rate = self.k2

            # Generate 2 children by swapping parents in argument of crossover operation
            children[individual] = self.crossover.order(
                self.crossover.uniform(self.population[individual]["chromosome"],
                                       self.population[individual + 1]["chromosome"]),
                self.population[individual + 1]["chromosome"])

            children[individual + 1] = self.crossover.order(
                self.crossover.uniform(self.population[individual + 1]["chromosome"],
                                       self.population[individual]["chromosome"]),
                self.population[individual]["chromosome"])

        return children

    def do_mutation(self, children) -> ndarray:
        """
        Handles the mutation
        param: children - empty 1D array
        return: children - 1D array of the mutated children holding chromosome information
        """

        for i in range(0, self.population_size):
            self.mutation.uniform(children[i])
            rand_num = random.random()
            if rand_num < 0.33:
                self.mutation.swap(children[i])
            elif 0.33 <= rand_num < 0.66:
                self.mutation.inversion(children[i])
            else:
                self.mutation.insertion(children[i])
        return children

    def log_configuration(self, individual):
        """
        Logs every interesting parameter
        param: chromosome - the best solution found
        """

        with open(f'../results/{self.__class__.__name__}/{self.TIMESTAMP}/best_chromosome.txt', 'a') as file:
            file.write(f'Population size: {self.population_size}'
                       f'\nGenerations: {self.max_generations}'
                       f'\nFitness scaling: {self.fitness_scaling.__name__}'
                       f'\nSelection method: {self.selection_method.__name__}'
                       f'\nAdaptive tournament size: {self.tournament_size}'
                       f'\nElitism: {self.elitism_percentage}'
                       f'\nBest fitness found: {individual["fitness"]:.2f}'
                       f'\nBest chromosome found: ')
            np.savetxt(file, individual["chromosome"], fmt='%d', newline=' ')
