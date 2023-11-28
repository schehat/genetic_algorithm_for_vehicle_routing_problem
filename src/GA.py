import os
import time
from concurrent.futures import ThreadPoolExecutor
from random import random
from typing import Callable, Tuple

import numpy as np
from numpy import ndarray

from mutation import Mutation
from enums import Purpose
from src.diversity_management import DiversityManagement
from src.education import Education
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
        TODO: BETWEEN SYNCHRONOUS AND PARALLEL CROSSOVER. STRAFEN AUCH ERHÖHEN NACH GENERATIONSZAHL
        TODO: Incorparate pattern improvement in normal mutation
    """

    TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    generation = 0
    diversify_counter = 0
    no_improvement_counter = 0
    MAX_RUNNING_TIME_IN_S = 3600 * 3
    start_time = None
    end_time = None
    children = None

    def __init__(self,
                 vrp_instance: VRPInstance,
                 population_size: int,
                 max_generations: int,
                 initial_population: Callable[[any], None],  # any => GA
                 fitness_scaling: Callable[[ndarray], ndarray],
                 selection_method: Callable[[ndarray, int], ndarray],
                 local_search_method,
                 distance_method,

                 tournament_size: int = 2,
                 tournament_size_increment: int = 1,
                 p_elite: float = 0.1,
                 elite_increment: int = 1,
                 p_c: float = 0.9,
                 p_m: float = 0.3,
                 p_education: float = 0.25,

                 p_adaptive_step: float = 0.1,
                 p_survivor_selection_step: float = 0.05,
                 p_selection_survival: float = 0.3,
                 p_diversify_step: float = 0.1,
                 p_diversify_survival: float = 0.3,

                 n_closest_neighbors: int = 3,
                 diversity_weight: float = 0.75,
                 capacity_penalty_factor: float = 10.0,
                 duration_penalty_factor: float = 5.0,
                 time_window_penalty: float = 10.0
                 ):

        self.vrp_instance: VRPInstance = vrp_instance
        self.population_size = population_size
        self.crossover = Crossover(self.vrp_instance)
        self.mutation = Mutation(self.vrp_instance)
        self.plotter = Plot(self)
        self.split = Split(self)
        self.education = Education(self)
        self.diversity_management = DiversityManagement(self)
        self.max_generations = max_generations
        self.initial_population = initial_population
        self.fitness_scaling: Callable[[ndarray], ndarray] = fitness_scaling
        self.selection_method: Callable[[ndarray, int], ndarray] = selection_method
        self.local_search_method = local_search_method

        self.tournament_size = tournament_size
        self.tournament_size_increment = tournament_size_increment
        self.n_elite = p_elite * self.population_size
        self.elite_increment = elite_increment
        self.p_c = p_c
        self.p_m = p_m
        self.p_education = p_education
        # self.p_repair = 0.5

        self.adaptive_step_size = p_adaptive_step * self.max_generations
        self.survivor_selection_step = p_survivor_selection_step * self.max_generations
        self.p_selection_survival = p_selection_survival
        self.threshold_no_improvement = int(0.5 * self.max_generations)
        self.diversify_step = p_diversify_step
        self.threshold_no_improvement_diversify = int(self.diversify_step * self.max_generations)
        self.p_diversify_survival = p_diversify_survival

        self.n_closest_neighbors = n_closest_neighbors
        self.diversity_weight = diversity_weight
        self.distance_method = distance_method
        # TODO: ADAPTIVE
        self.capacity_penalty_factor = capacity_penalty_factor
        self.duration_penalty_factor = duration_penalty_factor
        self.time_window_penalty = time_window_penalty

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
        self.route_data = []

    def run(self):
        """
        Execution of FISAGALS
        """

        self.start_time = time.time()
        self.initial_population(self)
        self.fitness_evaluation()
        self.diversity_management.calculate_biased_fitness()

        # Main GA loop
        for self.generation in range(self.max_generations):
            # Before starting the parent selection. Save percentage of best individuals
            top_individuals_i = np.argsort(self.population["fitness"])[
                                :int(self.population_size / self.n_elite)]
            top_individuals = self.population[top_individuals_i]

            self.selection_method(self.population, self.tournament_size)
            self.children = np.empty((self.population_size, self.vrp_instance.n_depots + self.vrp_instance.n_customers),
                                     dtype=int)

            # self.print_time_and_text("Before Crossover")
            self.do_crossover()
            # self.print_time_and_text("After Crossover")
            # self.population["chromosome"] = self.children
            # self.fitness_evaluation()

            for i, chromosome in enumerate(self.children):
                if random() <= self.p_m:
                    # if random() <= self.p_education:
                    #     try:
                    #         self.children[i], self.population[i]["fitness"] = self.education.run(chromosome, self.population[i]["fitness"])
                    #     except:
                    #         print("Education Error")
                    # else:
                    rand_num = random()
                    if rand_num < 0.33:
                        self.mutation.swap(self.children[i])
                    elif 0.33 <= rand_num < 0.66:
                        self.mutation.inversion(self.children[i])
                    else:
                        self.mutation.insertion(self.children[i])
            # self.print_time_and_text("After Education")

            # Replace old generation with new generation
            self.population["chromosome"] = self.children
            self.do_elitism(top_individuals)

            # self.fitness_scaling(self.population)
            self.fitness_evaluation()
            self.diversity_management.calculate_biased_fitness()
            self.save_fitness_statistics()

            # if (self.generation + 1) % self.adaptive_step_size == 0:
            #     self.tournament_size += self.tournament_size_increment
            #     self.n_elite += self.elite_increment

            # Track number of no improvements
            min_current_fitness = self.fitness_stats[self.generation]["min"]
            if min_current_fitness > self.best_solution["fitness"] - 0.0000001:
                self.no_improvement_counter += 1
                self.diversify_counter += 1
            else:
                self.no_improvement_counter = 0
                self.diversify_counter = 0
                self.best_solution = self.population[np.argmin(self.population["fitness"])].copy()

            if (self.generation + 1) % self.survivor_selection_step == 0:
                self.diversity_management.survivor_selection()

            # Diversify population
            if self.diversify_counter >= self.threshold_no_improvement_diversify:
                self.diversity_management.diversity_procedure()

            self.end_time = time.time()
            minutes, seconds = divmod(self.end_time - self.start_time, 60)
            print(
                f"Generation: {self.generation + 1}, Min/AVG Fitness: {self.fitness_stats[self.generation]['min']}/{self.fitness_stats[self.generation]['avg']}, Time: {int(minutes)}:{int(seconds)}")

            # Termination convergence criteria of GA
            if self.no_improvement_counter >= self.threshold_no_improvement or self.end_time - self.start_time >= self.MAX_RUNNING_TIME_IN_S:
                break

        print(f"min: {np.min(self.fitness_stats['min'])} ?= {self.best_solution}")
        self.local_search_method(self, self.best_solution)
        self.end_time = time.time()
        # Need to decode again to log chromosome correctly after local search
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
        # # self.local_search_method(self, self.population[0])
        # self.decode_chromosome(self.population[0]["chromosome"])
        # self.population[0]["fitness"] = self.total_fitness
        # self.population[0]["distance"] = self.total_distance
        # self.population[0]["time_warp"] = self.total_time_warp
        # self.population[0]["duration_violation"] = self.total_duration_violation
        # print(self.population[0])
        # self.log_configuration(self.population[0])

    def fitness_evaluation(self):
        for i, chromosome in enumerate(self.population["chromosome"]):
            total_fitness, total_distance, total_time_warp, total_duration_violation = self.decode_chromosome(
                chromosome)
            self.population[i]["fitness"] = total_fitness
            self.population[i]["distance"] = total_distance
            self.population[i]["time_warp"] = total_time_warp
            self.population[i]["duration_violation"] = total_duration_violation

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

    def do_crossover(self):
        for individual in range(0, self.population_size, 2):
            # self.do_adaptive_crossover_and_mutation_rate(individual)

            if random() <= self.p_c:
                # if random() < self.p_education:
                #     self.children[individual], self.population[individual][
                #     "fitness"] = self.crossover.periodic_crossover_with_insertions(
                #     self.population[individual]["chromosome"],
                #     self.population[individual + 1]["chromosome"], self, self.population[individual]["fitness"])
                #
                #     self.children[individual + 1], self.population[individual + 1][
                #         "fitness"] = self.crossover.periodic_crossover_with_insertions(
                #         self.population[individual + 1]["chromosome"],
                #         self.population[individual]["chromosome"], self, self.population[individual + 1]["fitness"])
                # else:
                # Generate 2 children by swapping parents in argument of crossover operation
                self.children[individual] = self.crossover.order_beginning(
                    self.population[individual]["chromosome"],
                    self.population[individual + 1]["chromosome"])

                self.children[individual + 1] = self.crossover.order_beginning(
                    self.population[individual + 1]["chromosome"],
                    self.population[individual]["chromosome"])
            else:
                self.children[individual] = self.population[individual]["chromosome"]
                self.children[individual + 1] = self.population[individual + 1]["chromosome"]

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

    def do_elitism(self, top_individuals: ndarray) -> None:
        """
        Perform elitism by replacing the worst individuals with the best individuals
        param: top_individuals - structured 3D array ["individual"]["chromosome]["fitness"]
        """

        worst_individuals_i = np.argsort(self.population["fitness"])[-int(self.population_size / self.n_elite):]
        self.population[worst_individuals_i] = top_individuals

    def print_time_and_text(self, text: str):
        self.end_time = time.time()
        minutes, seconds = divmod(self.end_time - self.start_time, 60)
        print(f"{text}: {int(minutes)}:{int(seconds)}")

    def save_fitness_statistics(self):
        self.fitness_stats[self.generation]["max"] = np.max(self.population["fitness"])
        self.fitness_stats[self.generation]["avg"] = np.mean(self.population["fitness"])
        self.fitness_stats[self.generation]["min"] = np.min(self.population["fitness"])

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
            file.write(f'Parameters:'
                       f'\npopulation_size: {self.population_size}, max_generations: {self.max_generations}, executed generations: {self.generation}, threshold_no_improvement: {self.threshold_no_improvement}'
                       f'\nfitness_scaling: {self.fitness_scaling.__name__}, selection_method: {self.selection_method.__name__}'
                       f'\np_c: {self.p_c}, p_m: {self.p_m}, NOT YET p_education: {self.p_education}'
                       f'\ntournament_size: {self.tournament_size}, n_elite: {self.n_elite}'
                       f'\nNOT YET adaptive_step_size: {self.adaptive_step_size}, tournament_size_increment: {self.tournament_size_increment}, elite_increment: {self.elite_increment}'
                       f'\nsurvivor_selection_step: {self.survivor_selection_step}, p_selection_survival: {self.p_selection_survival} '
                       f'\nthreshold_no_improvement_diversify: {self.threshold_no_improvement_diversify}, diversify_step: {self.diversify_step}, p_diversify_survival: {self.p_diversify_survival}'
                       f'\nn_closest_neighbors: {self.n_closest_neighbors}, diversity_weight: {self.diversity_weight}, distance_method: {self.distance_method.__name__}'
                       f'\ncapacity_penalty_factor: {self.capacity_penalty_factor}, duration_penalty_factor {self.duration_penalty_factor}, time_window_penalty: {self.time_window_penalty}'
                       f'\nBest fitness before/after local search: {np.min(self.fitness_stats["min"][self.fitness_stats["min"] != 0])} / {individual["fitness"]}'
                       f'\nBest individual found: {individual}'
                       f'\nTotal Runtime in seconds: {self.end_time - self.start_time}'
                       f'\n\nFitness stats min: {self.fitness_stats["min"][:self.generation + 1]} '
                       f'\n\nFitness stats avg: {self.fitness_stats["avg"][:self.generation + 1]} '
                       f'\nSolution Description: '
                       f'\np: {self.p_complete} '
                       f'\npred: {self.pred_complete} '
                       f'\ndistance {self.distance_complete}'
                       f'\nload: {self.capacity_complete} '
                       f'\ntime: {self.time_complete} '
                       f'\ntime warps: {self.time_warp_complete} '
                       f'\nduration: {self.duration_complete}'
                       f'\n\nAll individuals: {self.population}')
