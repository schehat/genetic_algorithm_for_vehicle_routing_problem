import os
import time
from concurrent.futures import ThreadPoolExecutor
from random import random
from typing import Callable, Tuple

import numpy as np
from numpy import ndarray

from mutation import Mutation
from enums import Purpose
from src.distance_measurement import broken_pairs_distance
from src.education import Education
from src.initial_population import initial_population_random
from src.split import Split
from vrp import Customer, Depot, VRPInstance
from crossover import Crossover
from plot import Plot
import datetime


class GA:
    """
    - Hybrid genetic algorithm with heuristics and local search
    - Chromosome representation specific integer string consisting of two parts:
        1. Number of customers for each depot
        2. The order of customers for each vehicle to serve
        E.g. for 2 depots and 7 customers (5, 2, 1, 2, 3, 4, 5, 6, 7)
        => first depot (index 0) has 5 customers, serving customers 1 - 5
        => second depot (Index 1) has 2 customers, serving customer 6 and 7
        TODO: MIX BETWEEN RANDOM AND LOCAL SEARCH? BIG POPULATION AND RANDOM THEN CUT AND HARD LOCAL SEARCH. SWITCH
        BETWEEN SYNCHRONOUS AND PARALLEL CROSSOVER. STRAFEN AUCH ERHÖHEN NACH GENERATIONSZAHL
    """

    TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    generation = 0
    num_generation_no_improvement = 0
    diversity_increment = 0
    NUM_GENERATIONS_NO_IMPROVEMENT_LIMIT = None
    NUM_GENERATIONS_DIVERSITY = None
    MAX_RUNNING_TIME_IN_S = 2700
    start_time = None
    end_time = None

    def __init__(self, vrp_instance: VRPInstance,
                 population_size: int,
                 max_generations: int,
                 initial_population: Callable[[any], None],  # any => GA
                 fitness_scaling: Callable[[ndarray], ndarray],
                 selection_method: Callable[[ndarray, int], ndarray],
                 local_search_complete,
                 tournament_size: int = 2,
                 tournament_size_increment: int = 1,
                 p_elite: float = 0.1,
                 elite_increment: int = 1,
                 p_c: float = 0.9,
                 p_m: float = 0.3):

        self.vrp_instance: VRPInstance = vrp_instance
        self.population_size = population_size
        self.crossover = Crossover(self.vrp_instance)
        self.mutation = Mutation(self.vrp_instance)
        self.plotter = Plot(self)
        self.split = Split(self)
        self.education = Education(self)
        self.max_generations = max_generations
        self.initial_population = initial_population
        self.fitness_scaling: Callable[[ndarray], ndarray] = fitness_scaling
        self.selection_method: Callable[[ndarray, int], ndarray] = selection_method
        self.local_search_complete = local_search_complete

        self.tournament_size = tournament_size
        self.tournament_size_increment = tournament_size_increment
        self.adaptive_step_size = 0.1 * self.max_generations
        self.n_elite = p_elite * self.population_size
        self.elite_increment = elite_increment
        self.p_c = p_c
        self.p_m = p_m
        self.p_education = 0.25
        self.p_repair = 0.5

        self.NUM_GENERATIONS_NO_IMPROVEMENT_LIMIT = self.max_generations * 0.5
        self.NUM_GENERATIONS_DIVERSITY = 0.1 * self.max_generations

        self.p_complete = np.array([], dtype=int)
        self.pred_complete = np.array([], dtype=int)
        self.distance_complete = np.array([], dtype=int)
        self.capacity_complete = np.array([], dtype=int)
        self.time_complete = np.array([], dtype=int)
        self.time_warp_complete = np.array([], dtype=int)
        self.duration_complete = np.array([], dtype=int)

        population_type = np.dtype([
            ("individual", int),
            ("chromosome", int,
             (self.vrp_instance.n_depots + self.vrp_instance.n_customers,)),
            ("fitness", float),
            ("distance", float),
            ("time_warp", float),
            ("duration_violation", float),
            ("diversity_contribution", float),
            ("fitness_ranked", int),
            ("diversity_contribution_ranked", int),
            ("biased_fitness", float)
        ])
        self.population = np.zeros(self.population_size, dtype=population_type)
        self.best_solution = np.zeros(1, dtype=population_type)
        self.best_solution[0]["fitness"] = float("inf")
        self.fitness_stats = np.zeros(max_generations, dtype=np.dtype([("max", float), ("avg", float), ("min", float),
                                                                       ("max_scaled", float), ("avg_scaled", float),
                                                                       ("min_scaled", float)]))

        self.n_closest_neighbors = 3
        self.diversity_weight = 0.5
        self.factor_diversity_survival = 0.5
        self.capacity_penalty_factor = 10
        self.duration_penalty_factor = 5
        self.time_window_penalty = 10
        self.route_data = []

    def run(self):
        """
        Execution of FISAGALS
        """

        self.start_time = time.time()
        self.initial_population(self)

        for self.generation in range(self.max_generations):
            # Fitness evaluation
            for i, chromosome in enumerate(self.population["chromosome"]):
                total_fitness, total_distance, total_time_warp, total_duration_violation = self.decode_chromosome(
                    chromosome)
                self.population[i]["fitness"] = total_fitness
                self.population[i]["distance"] = total_distance
                self.population[i]["time_warp"] = total_time_warp
                self.population[i]["duration_violation"] = total_duration_violation

            if self.generation > 0:
                self.do_elitism(top_individuals)
            self.calculate_biased_fitness()

            # self.fitness_scaling(self.population)

            # Increasing selection pressure over time by simple increment of parameters
            # if self.generation % self.adaptive_step_size == 0:
            #     self.tournament_size += self.tournament_size_increment
            #     self.n_elite += self.elite_increment

            # Before starting the parent selection. Save percentage of best individuals
            top_individuals_i = np.argsort(self.population["biased_fitness"])[
                                :int(self.population_size / self.n_elite)]
            top_individuals = self.population[top_individuals_i]
            self.selection_method(self.population, self.tournament_size)

            children = np.empty((self.population_size, self.vrp_instance.n_depots + self.vrp_instance.n_customers),
                                dtype=int)

            # Save statistics about raw fitness
            self.fitness_stats[self.generation]["max"] = np.max(self.population["fitness"])
            self.fitness_stats[self.generation]["avg"] = np.mean(self.population["fitness"])
            self.fitness_stats[self.generation]["min"] = np.min(self.population["fitness"])

            # Track number of no improvements
            min_fitness = self.fitness_stats[self.generation]["min"]
            if self.fitness_stats[self.generation]["min"] > self.best_solution["fitness"] - 0.0000001:
                self.num_generation_no_improvement += 1
                self.diversity_increment += 1
            else:
                self.num_generation_no_improvement = 0
                self.diversity_increment = 0

            # Check if there is a new best solution found
            if min_fitness < self.best_solution['fitness']:
                self.best_solution = self.population[np.argmin(self.population["fitness"])].copy()

            # Diversify population. TODO: SURVIVOR SELECTOR
            if self.NUM_GENERATIONS_DIVERSITY <= self.diversity_increment:
                print("DIVERSITY PROCEDURE")
                self.diversity_increment = 0
                for i, chromosome in enumerate(self.population["chromosome"]):
                    total_fitness, total_distance, total_time_warp, total_duration_violation = self.decode_chromosome(
                        chromosome)
                    self.population[i]["fitness"] = total_fitness
                    self.population[i]["distance"] = total_distance
                    self.population[i]["time_warp"] = total_time_warp
                    self.population[i]["duration_violation"] = total_duration_violation

                sorted_population = np.sort(self.population, order='fitness')
                # Determine the number of individuals to keep
                num_to_keep = int(self.factor_diversity_survival * len(self.population))
                # Extract the best individuals
                best_individuals = sorted_population[:num_to_keep].copy()
                initial_population_random(self)
                # Insert the best individuals into the new population
                self.population[:num_to_keep] = best_individuals

            # self.print_time_and_text("Before Crossover")
            children = self.do_crossover_synchronous(children)
            # self.print_time_and_text("After Crossover")

            for i, chromosome in enumerate(children):
                # if random() <= self.p_education:
                #     try:
                #         children[i], self.population[i]["fitness"] = self.education.run(chromosome, self.population[i]["fitness"])
                #     except:
                #         print("Education Error")
                if random() <= self.p_m:
                    rand_num = random()
                    if rand_num < 0.33:
                        self.mutation.swap(children[i])
                    elif 0.33 <= rand_num < 0.66:
                        self.mutation.inversion(children[i])
                    else:
                        self.mutation.insertion(children[i])
            # self.print_time_and_text("After Education")

            # Replace old generation with new generation
            self.population["chromosome"] = children

            self.end_time = time.time()
            minutes, seconds = divmod(self.end_time - self.start_time, 60)
            print(
                f"Generation: {self.generation + 1}, Min/AVG Fitness: {self.fitness_stats[self.generation]['min']}/{self.fitness_stats[self.generation]['avg']}, Time: {int(minutes)}:{int(seconds)}")

            # Termination convergence criteria of GA
            self.end_time = time.time()
            if self.num_generation_no_improvement >= self.NUM_GENERATIONS_NO_IMPROVEMENT_LIMIT or self.end_time - self.start_time >= self.MAX_RUNNING_TIME_IN_S:
                break

        print(f"min: {np.min(self.fitness_stats['min'])} ?= {self.best_solution}")
        self.local_search_complete(self, self.best_solution)
        self.end_time = time.time()
        # Need to decode again to log chromosome correctly after local search. TODO
        self.decode_chromosome(self.best_solution["chromosome"])
        print(f"min: {np.min(self.fitness_stats['min'])} ?= {self.best_solution}")
        if self.fitness_stats[self.generation]["min"] >= self.best_solution["fitness"]:
            self.fitness_stats[self.generation]["min"] = self.best_solution["fitness"]
        self.plotter.plot_fitness()
        self.plotter.plot_routes(self.best_solution["chromosome"])
        self.log_configuration(self.best_solution)

        # BELOW TESTING
        #
        # self.population[0]["chromosome"] = [16, 9, 14, 9,
        #                                     9, 42, 46, 39, 15, 25, 26, 23, 36, 32,
        #                                     35, 44, 31, 41, 7, 37,
        #
        #                                     34, 10, 45, 6, 27, 3, 48, 11,
        #                                     22,
        #
        #                                     28, 4, 19, 14, 1, 16,
        #                                     13, 33, 20, 29, 8, 5, 17, 18,
        #
        #                                     30,
        #                                     2, 47, 24, 12, 38, 40, 21, 43
        #                                     ]
        # # self.local_search_complete(self, self.population[0])
        # self.decode_chromosome(self.population[0]["chromosome"])
        # self.population[0]["fitness"] = self.total_fitness
        # self.population[0]["distance"] = self.total_distance
        # self.population[0]["time_warp"] = self.total_time_warp
        # self.population[0]["duration_violation"] = self.total_duration_violation
        # print(self.population[0])
        # self.log_configuration(self.population[0])

    def decode_chromosome(self, chromosome: ndarray, purpose: Purpose = Purpose.FITNESS) -> Tuple[
        ndarray, ndarray, ndarray, ndarray]:
        """
        Decoding chromosome by traversing the customers considering constraints and fetching the routes.
        Optional purpose for collecting routes for plotting
        param: chromosome - 1D array
        param: purpose - optional flag to collect routes
        """

        if purpose == Purpose.PLOTTING:
            self.route_data = []

        p_complete, pred_complete, distance_complete, capacity_complete, time_complete, time_warp_complete, duration_complete = self.split.split(
            chromosome)
        self.p_complete = p_complete
        self.pred_complete = pred_complete
        self.distance_complete = distance_complete
        self.capacity_complete = capacity_complete
        self.time_complete = time_complete
        self.time_warp_complete = time_warp_complete
        self.duration_complete = duration_complete

        i_route = 0

        i_depot = -1
        depot = None

        # Will hold the index where the depot routes end to aggregate distance and overtime
        depot_value_index = []

        for i_customer in chromosome[self.vrp_instance.n_depots:]:
            # Check if iterated through all routes of a depot then update depot
            if p_complete[i_route] == 0:
                if purpose == Purpose.PLOTTING:
                    # Exclude first iteration
                    if depot is not None:
                        self.collect_routes(depot, i_depot)

                depot_value_index.append(i_route - 1)

                i_route += 1
                i_depot += 1
                depot = self.vrp_instance.depots[i_depot]

                if purpose == Purpose.PLOTTING:
                    self.collect_routes(depot, i_depot)

            pred = pred_complete[i_route]
            # Then single route complete back to depot
            if pred != pred_complete[i_route - 1]:
                self.collect_routes(depot, i_depot)
                depot_value_index.append(i_route - 1)

            customer1: Customer = self.vrp_instance.customers[i_customer - 1]

            if purpose == Purpose.PLOTTING:
                self.collect_routes(customer1, i_depot)

            i_route += 1

        # At the end last depot
        if purpose == Purpose.PLOTTING:
            self.collect_routes(depot, i_depot)

        depot_value_index = np.array(depot_value_index)
        # Select the values at depot_value_index
        selected_values = distance_complete[np.concatenate([depot_value_index])]
        total_distance = np.sum(selected_values)

        selected_values = time_warp_complete[np.concatenate([depot_value_index])]
        total_time_warp = np.sum(selected_values)

        selected_values = duration_complete[np.concatenate([depot_value_index])]
        exceeding_values = selected_values[selected_values > self.vrp_instance.max_duration_of_a_route]
        differences = exceeding_values - self.vrp_instance.max_duration_of_a_route
        total_duration_violation = np.sum(differences)

        zero_indices = np.where(p_complete == 0)[0]
        selected_values = p_complete[np.concatenate([zero_indices - 1])]
        total_fitness = np.sum(selected_values)

        return total_fitness, total_distance, total_time_warp, total_duration_violation

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

    def calculate_biased_fitness(self):
        self.calculate_diversity_contribution()

        fitness_values = self.population["fitness"]
        diversity_control_values = self.population["diversity_contribution"]

        # Calculate the ranks
        fitness_indexes = np.argsort(fitness_values)
        # Higher distance measurement for diversity contributes to lower ranks. Therefore, reverse array
        diversity_contribution_indexes = np.argsort(diversity_control_values)[::-1]

        # Initialize arrays to store the ranked values
        fitness_ranked = np.zeros_like(fitness_indexes)
        diversity_contribution_ranked = np.zeros_like(diversity_contribution_indexes)

        # Assign ranks to fitness_ranked and diversity_contribution_ranked. Start ranks from 1
        fitness_ranked[fitness_indexes] = np.arange(1, len(fitness_indexes) + 1)
        diversity_contribution_ranked[diversity_contribution_indexes] = np.arange(1,
                                                                                  len(diversity_contribution_indexes) + 1)

        # Now you can use fitness_ranked and diversity_contribution_ranked to calculate biased_fitness
        biased_fitness = fitness_ranked + self.diversity_weight * diversity_contribution_ranked

        # Update the population array with the new values
        self.population["fitness_ranked"] = fitness_ranked
        self.population["diversity_contribution_ranked"] = diversity_contribution_ranked
        self.population["biased_fitness"] = biased_fitness

    def calculate_diversity_contribution(self):
        for i, chromosome_a in enumerate(self.population["chromosome"]):
            distances = []
            for j, chromosome_b in enumerate(self.population["chromosome"]):
                # Avoid calculating distance with the same chromosome
                if i != j:
                    distance = broken_pairs_distance(chromosome_a, chromosome_b, self.vrp_instance.n_depots)
                    distances.append((j, distance))

            # Sort distances and pick n_closest_neighbors
            distances.sort(key=lambda x: x[1])
            n_closest_neighbors = distances[:self.n_closest_neighbors]
            # Calculate the average distance of the n_closest_neighbors
            avg_distance = np.mean([dist for _, dist in n_closest_neighbors])
            # Set diversity_contribution
            self.population[i]["diversity_contribution"] = avg_distance

    def do_elitism(self, top_individuals: ndarray) -> None:
        """
        Perform elitism by replacing the worst individuals with the best individuals
        param: top_individuals - structured 3D array ["individual"]["chromosome]["fitness"]
        """

        worst_individuals_i = np.argsort(self.population["biased_fitness"])[
                              :int(self.population_size / self.n_elite)]
        self.population[worst_individuals_i] = top_individuals

    def do_crossover_asynchronous(self, children: np.ndarray) -> np.ndarray:
        """
        Handles the crossover
        param: children - empty 1D array
        return: children - 1D array of the generation holding chromosome information
        """

        def crossover_for_individual(individual):
            if random() <= self.p_c:
                # Generate 2 children by swapping parents in the crossover operation
                child, fitness = self.crossover.periodic_crossover_with_insertions(
                    self.population[individual]["chromosome"],
                    self.population[individual + 1]["chromosome"],
                    self, self.population[individual]["fitness"])
            else:
                child = self.population[individual]["chromosome"]
                fitness = self.population[individual]["fitness"]

            return child, fitness

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(crossover_for_individual, range(0, self.population_size, 2)))

        # Unpack the results
        for i, (child, fitness) in enumerate(results):
            children[i * 2] = child
            self.population[i * 2]["fitness"] = fitness

            children[i * 2 + 1] = self.population[i * 2 + 1]["chromosome"]
            self.population[i * 2 + 1]["fitness"] = self.population[i * 2 + 1]["fitness"]

        return children

    def do_crossover_synchronous(self, children: ndarray) -> ndarray:
        """
        Handles the crossover
        param: children - empty 1D array
        return: children - 1D array of the generation holding chromosome information
        """

        for individual in range(0, self.population_size, 2):
            # self.do_adaptive_crossover_and_mutation_rate(individual)

            if random() <= self.p_c:
                # Generate 2 children by swapping parents in argument of crossover operation
                children[individual] = self.crossover.order_beginning(
                    self.population[individual]["chromosome"],
                    self.population[individual + 1]["chromosome"])

                children[individual + 1] = self.crossover.order_beginning(
                    self.population[individual + 1]["chromosome"],
                    self.population[individual]["chromosome"])
            else:
                children[individual] = self.population[individual]["chromosome"]
                children[individual + 1] = self.population[individual + 1]["chromosome"]

        return children

    def do_crossover_pix_synchronous(self, children: ndarray) -> ndarray:
        """
        Handles the crossover
        param: children - empty 1D array
        return: children - 1D array of the generation holding chromosome information
        """

        for individual in range(0, self.population_size, 2):
            # self.do_adaptive_crossover_and_mutation_rate(individual)

            if random() <= self.p_c:
                # Generate 2 children by swapping parents in argument of crossover operation
                children[individual], self.population[individual][
                    "fitness"] = self.crossover.periodic_crossover_with_insertions(
                    self.population[individual]["chromosome"],
                    self.population[individual + 1]["chromosome"], self, self.population[individual]["fitness"])

                children[individual + 1], self.population[individual + 1][
                    "fitness"] = self.crossover.periodic_crossover_with_insertions(
                    self.population[individual + 1]["chromosome"],
                    self.population[individual]["chromosome"], self, self.population[individual + 1]["fitness"])
            else:
                children[individual] = self.population[individual]["chromosome"]
                children[individual + 1] = self.population[individual + 1]["chromosome"]

        return children

    def do_adaptive_crossover_and_mutation_rate(self, individual: int):
        # Adaptive rates for genetic operators
        min_parent_fitness = min(self.population[individual]["fitness"],
                                 self.population[individual + 1]["fitness"])
        if min_parent_fitness >= self.fitness_stats[self.generation]["avg_scaled"]:
            max_parent_fitness = max(self.population[individual]["fitness"],
                                     self.population[individual + 1]["fitness"])
            numerator = min_parent_fitness - self.fitness_stats[self.generation]["min_scaled"]
            denominator = max_parent_fitness - self.fitness_stats[self.generation]["min_scaled"]

            try:
                self.crossover.adaptive_crossover_rate = self.p_c * (numerator / denominator)
                self.mutation.adaptive_mutation_rate = self.p_m * (numerator / denominator)
            except ZeroDivisionError:
                self.crossover.adaptive_crossover_rate = self.p_c
                self.mutation.adaptive_mutation_rate = self.p_m
        else:
            self.crossover.adaptive_crossover_rate = self.p_c
            self.mutation.adaptive_mutation_rate = self.p_m

    def do_mutation(self, children) -> ndarray:
        """
        Handles the mutation
        param: children - empty 1D array
        return: children - 1D array of the mutated children holding chromosome information
        """

        for i in range(0, self.population_size):
            self.mutation.uniform(children[i])
            if random() <= self.p_m:
                rand_num = random()
                if rand_num < 0.33:
                    self.mutation.swap(children[i])
                elif 0.33 <= rand_num < 0.66:
                    self.mutation.inversion(children[i])
                else:
                    self.mutation.insertion(children[i])

        return children

    def print_time_and_text(self, text: str):
        self.end_time = time.time()
        minutes, seconds = divmod(self.end_time - self.start_time, 60)
        print(f"{text}: {int(minutes)}:{int(seconds)}")

    def log_configuration(self, individual) -> None:
        """
        Logs every relevant parameter
        param: individual - the best solution found
        """

        # Sort population
        sorted_indices = np.argsort(self.population["fitness"])
        self.population[:] = self.population[sorted_indices]

        directory = f'../results/{self.__class__.__name__}/{self.TIMESTAMP}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, 'best_chromosome.txt'), 'a') as file:
            file.write(f'Population size: {self.population_size}'
                       f'\nMax generations: {self.max_generations}'
                       f'\nExecuted generations: {self.generation}'
                       f'\nFitness scaling: {self.fitness_scaling.__name__}'
                       f'\nSelection method: {self.selection_method.__name__}'
                       f'\nAdaptive tournament size: {self.tournament_size}'
                       f'\nElitism: {self.n_elite}'
                       f'\nBest fitness found before local search: {np.min(self.fitness_stats["min"][self.fitness_stats["min"] != 0])}'
                       f'\nBest fitness found after local search: {individual["fitness"]:.2f}'
                       f'\nBest individual found: {individual}'
                       f'\nTotal Runtime in seconds: {self.end_time - self.start_time}'
                       f'\nFitness stats min: {self.fitness_stats["min"][:self.generation + 1]} '
                       f'\nSolution Description: '
                       f'\np: {self.p_complete} '
                       f'\npred: {self.pred_complete} '
                       f'\ndistance {self.distance_complete}'
                       f'\nload: {self.capacity_complete} '
                       f'\ntime: {self.time_complete} '
                       f'\ntime warps: {self.time_warp_complete} '
                       f'\nduration: {self.duration_complete}'
                       f'\n\nAll individuals: {self.population}')
