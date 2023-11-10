import random
from typing import Callable

import numpy as np
from numpy import ndarray

from crossover import Crossover
from mutation import Mutation
from enums import Purpose
from vrp import Customer, Depot, VRPInstance
import datetime


class GA1:
    """
    - Hybrid genetic algorithm with heuristics and local search
    - Chromosome representation specific integer string consisting of two parts:
        1. Number of customers for each depot
        2. The order of customers for each vehicle to serve
        E.g. for 2 depots and 7 customers (5, 2, 1, 2, 3, 4, 5, 6, 7)
        => first depot (index 0) has 5 customers, serving customers 1 - 5
        => second depot (Index 1) has 2 customers, serving customer 6 and 7
    """

    THRESHOLD = 200
    TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    generation = 0

    def __init__(self, vrp_instance: VRPInstance,
                 population_size: int,
                 crossover_rate: float,
                 mutation_rate: float,
                 max_generations: int,
                 initial_population: Callable[[any], None],  # any => GA
                 fitness_scaling: Callable[[ndarray], ndarray],
                 selection_method: Callable[[ndarray, int], ndarray],
                 local_search_complete,
                 local_search_single,
                 tournament_size: int = 5,
                 tournament_size_increment: int = 1,
                 elitism_percentage: float = 0.1,
                 k1: float = 0.9,
                 k2: float = 0.3):
        """
            param: k1 and k2 - control rates for adaptive genetic operators
        """

        self.vrp_instance: VRPInstance = vrp_instance
        self.population_size = population_size
        self.crossover = Crossover(self.vrp_instance, crossover_rate)
        self.mutation = Mutation(self.vrp_instance, mutation_rate)
        self.max_generations = max_generations
        self.initial_population = initial_population
        self.fitness_scaling: Callable[[ndarray], ndarray] = fitness_scaling
        self.selection_method: Callable[[ndarray, int], ndarray] = selection_method
        self.local_search_complete = local_search_complete
        self.local_search_single = local_search_single
        self.tournament_size = tournament_size
        self.tournament_size_increment = tournament_size_increment
        self.elitism_percentage = elitism_percentage
        self.k1 = k1
        self.k2 = k2

        population_type = np.dtype([
            ("individual", int),
            ("chromosome", int,
             (self.vrp_instance.n_depots + self.vrp_instance.n_customers,)),
            ("fitness", float),
            ("distance", float),
            ("timeout", float)
        ])
        self.population = np.zeros(self.population_size, dtype=population_type)
        self.best_solution = np.zeros(1, dtype=population_type)
        self.best_solution[0]["fitness"] = float("inf")
        # Note after scaling order of fitness reversed: min fitness mapped to max_scaled

        self.fitness_stats = np.zeros(max_generations, dtype=np.dtype([("max", float), ("avg", float), ("min", float),
                                                                       ("max_scaled", float), ("avg_scaled", float),
                                                                       ("min_scaled", float)]))
        # Total for one individual not the total of all individuals
        self.total_fitness = 0.0
        self.total_distance = 0.0
        self.total_timeout = 0

        self.route_data = []

    def run(self):
        """
        Execution of FISAGALS
        """

        self.initial_population(self)

        for self.generation in range(self.max_generations):
            # Fitness evaluation
            for i, chromosome in enumerate(self.population["chromosome"]):
                self.decode_chromosome(chromosome, Purpose.FITNESS)
                self.population[i]["fitness"] = self.total_fitness
                self.population[i]["distance"] = self.total_distance
                self.population[i]["timeout"] = self.total_timeout

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
                self.tournament_size += self.tournament_size_increment
            self.selection_method(self.population, self.tournament_size)
            self.do_elitism(top_individuals)

            children = np.empty((self.population_size, self.vrp_instance.n_depots + self.vrp_instance.n_customers),
                                dtype=int)

            children = self.do_crossover(children)
            children = self.do_mutation(children)
            # self.local_search_single(self, self.best_solution)

            # Replace old generation with new generation
            self.population["chromosome"] = children

            # Termination convergence criteria
            print(f"{self.generation}")
            # fitness_bound = self.fitness_stats["min"][generation - int(self.max_generations*0.3)] if generation - int(self.max_generations*0.3) >= 0 else float('inf')
            # if self.fitness_stats["min"][generation] - self.THRESHOLD > fitness_bound:
            #     break

        # # get best individual
        # self.local_search_complete(self, self.best_solution)
        # print(f"min: {np.min(self.fitness_stats['min'])} ?= {self.best_solution}")
        # if self.best_solution["fitness"] < np.min(self.fitness_stats["min"]):
        #     self.fitness_stats[self.max_generations - 1]["min"] = self.best_solution["fitness"]

        # plot_fitness(self)
        # plot_routes(self, self.best_solution["chromosome"])
        # self.log_configuration(self.best_solution)

        self.population[0]["chromosome"] = [14, 19, 8, 9,
                                            44, 45, 33, 15, 37, 17,
                                            42, 19, 40, 41, 13,
                                            25, 18, 4,

                                            6, 27, 1, 32, 11, 46,
                                            48, 8, 26, 31, 28, 22,
                                            23, 7, 43, 24, 14,
                                            12, 47,

                                            9, 34, 30, 39, 10,
                                            49, 5, 38,

                                            35, 36, 3, 20,
                                            21, 50, 16, 2, 29
                                            ]
        self.split(self.population[0]["chromosome"])
        # plot_routes(self, self.population[0]["chromosome"])
        self.decode_chromosome(self.population[0]["chromosome"], Purpose.FITNESS)
        self.population[0]["fitness"] = self.total_fitness
        self.population[0]["distance"] = self.total_distance
        self.population[0]["timeout"] = self.total_timeout
        print(self.population[0])

    def decode_chromosome(self, chromosome: ndarray, purpose: Purpose) -> None:
        """
        TODO: smarteres splitten
        Decoding chromosome by traversing the genes considering constraints and fetching the routes.
        Expecting a purpose to evaluate which operation should be used
        param: chromosome - 1D array
        param: purpose - defines which operation should be used
        """

        customer_index = self.vrp_instance.n_depots
        self.total_fitness = 0.0
        self.total_distance = 0.0
        self.total_timeout = 0

        for depot_index in range(self.vrp_instance.n_depots):
            depot_i_n_customers = chromosome[depot_index]
            # Capacity for every vehicle the same at the moment. TODO dynamic capacity with vehicle class
            vehicle_i_capacity = 0
            # TODO route duration is not travelled distance!!!
            vehicle_i_travelled_distance = 0
            vehicle_i_current_time = 0
            vehicle_i_depot: Depot = self.vrp_instance.depots[depot_index]

            for j in range(depot_i_n_customers):
                customer_value1 = chromosome[customer_index + j]
                # Indexing of customers starts with 1 not 0, so -1 necessary
                customer1: Customer = self.vrp_instance.customers[customer_value1 - 1]

                # First iteration in loop: first trip 
                if j == 0:
                    # Add distance from depot to customer with the euclidean distance
                    # Assuming single customer demand <= vehicle max capacity
                    # TODO add capacity constraint meaning vehicles with different capacity
                    # Thus customer demand > vehicle max capacity possible but at least 1 vehicle exists with greater capacity
                    vehicle_i_capacity += customer1.demand

                    # Track travelled distance in total and per vehicle to check route duration constraint
                    distance = self.euclidean_distance(vehicle_i_depot, customer1)
                    self.total_distance += distance
                    vehicle_i_travelled_distance += distance

                    # At the beginning vehicle time starts always with the first customer start window
                    vehicle_i_current_time = customer1.start_time_window

                    if purpose == purpose.PLOTTING:
                        self.collect_routes(vehicle_i_depot, depot_index)
                        self.collect_routes(customer1, depot_index)

                # Check if next customer exists in route
                if j < depot_i_n_customers - 1:
                    customer_value2 = chromosome[customer_index + j + 1]
                    customer2: Customer = self.vrp_instance.customers[customer_value2 - 1]

                    # Check customer 2 demand exceeds vehicle capacity limit
                    # TODO Add heterogeneous capacity for vehicles
                    if vehicle_i_capacity + customer2.demand > self.vrp_instance.max_capacity:
                        # Trip back to depot necessary. Assuming heading back to same depot it came from
                        # TODO visit different depot if possible e.g. AF-VRP charging points for robots

                        # From customer 1 to depot
                        distance1 = self.euclidean_distance(customer1, vehicle_i_depot)
                        self.total_distance += distance1
                        vehicle_i_travelled_distance += distance1

                        # TODO LOG info vehicle_i travel and capacity... Learn to use all vehicles

                        # New vehicle from depot to customer 2
                        distance2 = self.euclidean_distance(vehicle_i_depot, customer2)
                        self.total_distance += distance2
                        # Reset values. Capacity for customer2.demand added later
                        vehicle_i_capacity = 0
                        vehicle_i_travelled_distance = distance2
                        vehicle_i_current_time = customer2.start_time_window

                        if purpose == purpose.PLOTTING:
                            self.collect_routes(vehicle_i_depot, depot_index)
                            self.collect_routes(customer2, depot_index)
                    else:
                        # Add distance between customer 1 and customer 2
                        distance = self.euclidean_distance(customer1, customer2)
                        self.total_distance += distance
                        vehicle_i_travelled_distance += distance

                        vehicle_i_current_time += customer1.service_duration + distance

                        if purpose == purpose.PLOTTING:
                            self.collect_routes(customer2, depot_index)

                    vehicle_i_capacity += customer2.demand
                    # Check if vehicle reaches customer 2 before start window then needs to wait
                    if vehicle_i_current_time < customer2.start_time_window:
                        vehicle_i_current_time = customer2.start_time_window
                    # Check if vehicle reaches customer 2 later than end time window then penalty
                    elif vehicle_i_current_time > customer2.end_time_window:
                        self.total_timeout += vehicle_i_current_time - customer2.start_time_window

                # Last iteration in loop, add trip from last customer to depot
                if j >= depot_i_n_customers - 1:
                    # Add distance between customer 1 and customer 2
                    distance = self.euclidean_distance(customer1, vehicle_i_depot)
                    self.total_distance += distance
                    vehicle_i_travelled_distance += distance

                    if purpose == purpose.PLOTTING:
                        self.collect_routes(vehicle_i_depot, depot_index)

            customer_index += depot_i_n_customers

        # simple fitness evaluation
        self.total_fitness = self.total_distance  # + self.total_timeout**2

    def split(self, chromosome: ndarray) -> None:
        customer_index = self.vrp_instance.n_depots

        for depot_index in range(self.vrp_instance.n_depots):
            depot_i_n_customers = chromosome[depot_index]
            # Capacity for every vehicle the same at the moment. TODO dynamic capacity with vehicle class
            load = 0
            # TODO route duration is not travelled distance!!!
            vehicle_i_depot: Depot = self.vrp_instance.depots[depot_index]

            p = np.full(depot_i_n_customers + 1, np.inf)
            p[0] = 0
            pred = np.zeros(depot_i_n_customers + 1)

            for t in range(depot_i_n_customers):
                load = 0
                i = t + 1

                customer_value_i = chromosome[customer_index + (i - 1)]
                # Indexing of customers starts with 1 not 0, so -1 necessary
                customer_i: Customer = self.vrp_instance.customers[customer_value_i - 1]

                while i <= depot_i_n_customers and load + customer_i.demand <= self.vrp_instance.max_capacity:
                    load += customer_i.demand

                    if i == t + 1:
                        distance = self.euclidean_distance(vehicle_i_depot, customer_i)
                        self.total_distance += distance
                        cost = distance
                    else:
                        customer_value_pre_i = chromosome[customer_index + (i - 1 - 1)]
                        customer_pre_i: Customer = self.vrp_instance.customers[customer_value_pre_i - 1]
                        distance = self.euclidean_distance(customer_pre_i, customer_i)
                        cost += distance

                    distance = self.euclidean_distance(customer_i, vehicle_i_depot)
                    if p[t] + cost + distance < p[i]:
                        p[i] = p[t] + cost + distance
                        pred[i] = t

                    i += 1
                    customer_value_i = chromosome[customer_index + (i-1)]
                    customer_i: Customer = self.vrp_instance.customers[customer_value_i - 1]

            customer_index += depot_i_n_customers

    @staticmethod
    def euclidean_distance(obj1, obj2) -> float:
        """
        Calculate fitness for a single chromosome
        param: obj1 and obj2 - Customers or Depots
        """

        return np.linalg.norm(
            np.array([obj1.x, obj1.y]) - np.array([obj2.x, obj2.y]))

    def collect_routes(self, obj, from_vehicle_i) -> None:
        """
        Add routing points and their id
        param: obj - Customer or Depot object
        param: from_vehicle_i - index from which vehicle the route is coming from
        """

        # Create empty entry for the vehicle
        while len(self.route_data) <= from_vehicle_i:
            self.route_data.append({'x_pos': [], 'y_pos': [], 'customer_ids': []})

        self.route_data[from_vehicle_i]['x_pos'].append(obj.x)
        self.route_data[from_vehicle_i]['y_pos'].append(obj.y)

        # Differentiate between Customer or Depot object
        if type(obj) is Customer:
            self.route_data[from_vehicle_i]['customer_ids'].append(f"C{obj.id}")
        elif type(obj) is Depot:
            # Blind label
            self.route_data[from_vehicle_i]['customer_ids'].append("")
        else:
            print("ERROR: unexpected behavior")

    def do_elitism(self, top_individuals: ndarray) -> None:
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

        self.crossover.adaptive_crossover_rate = self.k1
        self.mutation.adaptive_mutation_rate = self.k2
        for individual in range(0, self.population_size, 2):
            #     # Adaptive rates for genetic operators
            #     min_parent_fitness = min(self.population[individual]["fitness"],
            #                              self.population[individual + 1]["fitness"])
            #     if min_parent_fitness >= self.fitness_stats[self.generation]["avg_scaled"]:
            #         max_parent_fitness = max(self.population[individual]["fitness"],
            #                                  self.population[individual + 1]["fitness"])
            #         numerator = min_parent_fitness - self.fitness_stats[self.generation]["min_scaled"]
            #         denominator = max_parent_fitness - self.fitness_stats[self.generation]["min_scaled"]
            #
            #         try:
            #             self.crossover.adaptive_crossover_rate = self.k1 * (numerator / denominator)
            #             self.mutation.adaptive_mutation_rate = self.k2 * (numerator / denominator)
            #         except ZeroDivisionError:
            #             self.crossover.adaptive_crossover_rate = self.k1
            #             self.mutation.adaptive_mutation_rate = self.k2
            #     else:
            #         self.crossover.adaptive_crossover_rate = self.k1
            #         self.mutation.adaptive_mutation_rate = self.k2

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

    def log_configuration(self, individual) -> None:
        """
        Logs every interesting parameter
        param: chromosome - the best solution found
        """

        # Sort population
        sorted_indices = np.argsort(self.population["fitness"])
        self.population[:] = self.population[sorted_indices]

        with open(f'../results/{self.__class__.__name__}/{self.TIMESTAMP}/best_chromosome.txt', 'a') as file:
            file.write(f'Population size: {self.population_size}'
                       f'\nGenerations: {self.max_generations}'
                       f'\nFitness scaling: {self.fitness_scaling.__name__}'
                       f'\nSelection method: {self.selection_method.__name__}'
                       f'\nAdaptive tournament size: {self.tournament_size}'
                       f'\nElitism: {self.elitism_percentage}'
                       f'\nBest fitness found before local search: {np.min(self.fitness_stats["min"])}'
                       f'\nBest fitness found after local search: {individual["fitness"]:.2f}'
                       f'\nBest individual found: {individual}'
                       f'\n\nAll individuals: {self.population}')
            # np.savetxt(file, individual["chromosome"], fmt='%d', newline=' ')


from plot import plot_fitness, plot_routes
