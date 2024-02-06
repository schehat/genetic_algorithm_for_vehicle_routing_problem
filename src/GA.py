import os
import time
from random import random, shuffle
from typing import Callable, Tuple

import numpy as np
from numpy import ndarray

from mutation import Mutation
from enums import Purpose
from src.distance_measurement import EuclideanDistance
from src.diversity_management import DiversityManagement
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
    """

    TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    generation = 0
    diversify_counter = 0
    no_improvement_counter = 0
    MAX_RUNNING_TIME_IN_S = 3600 * 1.0
    start_time = None
    end_time = None
    children = None

    def __init__(self,
                 vrp_instance: VRPInstance,
                 population_size: int,
                 max_generations: int,
                 initial_population: Callable[[any], None],  # any => GA
                 fitness_scaling: Callable[[ndarray], ndarray],
                 selection_method: Callable[[ndarray, int, str], ndarray],
                 local_search_method,
                 distance_method,
                 problem_type,
                 file_prefix_name,
                 hybrid,

                 tournament_size: int = 2,
                 n_elite: int = 10,
                 p_c: float = 0.8,
                 p_m: float = 0.4,

                 penalty_step: int = 2,
                 survivor_selection_step: int = 10,
                 p_selection_survival: float = 0.7,
                 kill_clone_step: int = 100,
                 diversify_step: float = 300,
                 p_diversify_survival: float = 0.3,

                 n_closest_neighbors: int = 3,
                 diversity_weight: float = 0.75,
                 capacity_penalty_factor: float = 5.0,
                 duration_penalty_factor: float = 5.0,
                 time_window_penalty: float = 5.0,
                 penalty_factor: float = 0.05,

                 target_feasible_proportion: float = 0.25
                 ):

        self.vrp_instance: VRPInstance = vrp_instance
        self.population_size = population_size
        self.crossover = Crossover(self.vrp_instance)
        self.mutation = Mutation(self.vrp_instance)

        self.file_prefix_name = file_prefix_name
        self.problem_type = problem_type
        self.plotter = Plot(self)
        self.split = Split(self)
        self.education = Education(self)
        self.diversity_management = DiversityManagement(self)
        self.euclidean_distance = EuclideanDistance(self)
        self.max_generations = max_generations
        self.initial_population = initial_population
        self.fitness_scaling: Callable[[ndarray], ndarray] = fitness_scaling
        self.selection_method: Callable[[ndarray, int], ndarray] = selection_method
        self.local_search_method = local_search_method
        self.hybrid = hybrid

        self.tournament_size = tournament_size
        self.n_elite = n_elite
        self.p_c = p_c
        self.p_m = p_m

        self.penalty_step = penalty_step
        self.survivor_selection_step = survivor_selection_step
        self.p_selection_survival = p_selection_survival
        self.kill_clone_step = kill_clone_step
        self.threshold_no_improvement = int(0.5 * self.max_generations)
        self.diversify_step = diversify_step
        self.p_diversify_survival = p_diversify_survival

        self.n_closest_neighbors = n_closest_neighbors
        self.diversity_weight = diversity_weight
        self.distance_method = distance_method
        self.capacity_penalty_factor = capacity_penalty_factor
        self.duration_penalty_factor = duration_penalty_factor
        self.time_window_penalty = time_window_penalty
        self.penalty_factor = penalty_factor

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
            ("capacity_violation", float),
            ("time_warp", float),
            ("duration_violation", float),
            ("diversity_contribution", float),
            ("fitness_ranked", int),
            ("diversity_contribution_ranked", int),
            ("biased_fitness", float)
        ])
        self.population = np.zeros(self.population_size, dtype=population_type)
        self.target_feasible_proportion = int(target_feasible_proportion * self.population_size)
        self.n_feasible = 0
        self.best_solution = np.zeros(1, dtype=population_type)
        self.best_solution[0]["fitness"] = float("inf")
        self.fitness_stats = np.zeros(max_generations + 1,
                                      dtype=np.dtype(
                                          [("max", float), ("avg", float), ("min", float), ("min_feasible", float),
                                           ("max_scaled", float), ("avg_scaled", float),
                                           ("min_scaled", float)]))
        self.feasible_stats = np.zeros(max_generations + 1, dtype=np.dtype([("feasible", float), ("infeasible", float)]))
        self.route_data = []

        self.education_old_fitness = 0

    def run(self):
        """
        Execution of FISAGALS
        """

        if self.hybrid:
            self.initial_population(self)  # heuristic
        else:
            initial_population_random(self, 0, self.population_size)
        self.fitness_evaluation()
        if self.hybrid:
            self.diversity_management.calculate_biased_fitness()

        # Main GA loop
        self.start_time = time.time()
        self.run_generations(self.generation, self.max_generations)

        condition = (self.population["capacity_violation"] == 0) & (self.population["time_warp"] == 0) & (
                self.population["duration_violation"] == 0)

        feasible_individuals = self.population[condition]
        best_feasible_solution = None
        if len(feasible_individuals) > 0:
            top_feasible_individual_i = np.argsort(feasible_individuals["fitness"])[0]
            top_feasible_individual = feasible_individuals[top_feasible_individual_i]
            best_feasible_solution = top_feasible_individual.copy()
            self.best_solution = top_feasible_individual.copy()

        old_best_solution = self.best_solution
        print(f"best solution before local search {self.best_solution['fitness']}")

        if self.hybrid:
            self.local_search_method(self, self.best_solution)

        self.end_time = time.time()

        # Need to decode again to log chromosome correctly after local search
        self.decode_chromosome(self.best_solution["chromosome"])
        print(f"best solution after local search {self.best_solution['fitness']}")
        print(f"best solution: {self.best_solution}")
        print(f"best feasible: {best_feasible_solution}")
        self.fitness_stats[self.generation]["min"] = self.best_solution["fitness"]
        if self.best_solution["capacity_violation"] == 0 and self.best_solution["time_warp"] == 0 and \
                self.best_solution["duration_violation"] == 0:
            self.fitness_stats[self.generation]["min_feasible"] = self.best_solution["fitness"]
        else:
            self.fitness_stats[self.generation]["min_feasible"] = best_feasible_solution[
                "fitness"] if best_feasible_solution is not None else 0

        self.plotter.plot_fitness()
        self.plotter.plot_routes(self.best_solution["chromosome"])
        self.log_configuration(self.best_solution, old_best_solution, best_feasible_solution)

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
        # total_fitness, total_distance, total_capacity_violation, total_time_warp, total_duration_violation = self.decode_chromosome(self.population[0]["chromosome"])
        # self.population[0]["fitness"] = total_fitness
        # self.population[0]["distance"] = total_distance
        # self.population[0]["capacity_violation"] = total_capacity_violation
        # self.population[0]["time_warp"] = total_time_warp
        # self.population[0]["duration_violation"] = total_duration_violation
        # print(self.population[0])
        # self.log_configuration(self.population[0])

    def run_generations(self, current_generation, max_generation):
        self.generation = current_generation
        self.max_generations = max_generation
        self.save_fitness_statistics()
        for self.generation in range(1, self.max_generations + 1):
            # Before starting the parent selection. Save percentage of best individuals
            condition = (self.population["capacity_violation"] == 0) & (self.population["time_warp"] == 0) & (
                    self.population["duration_violation"] == 0)

            feasible_individuals = self.population[condition]
            top_feasible_individuals_i = np.argsort(feasible_individuals["fitness"])[:self.n_elite - 1]
            top_feasible_individuals = feasible_individuals[top_feasible_individuals_i]

            # Inverse condition
            infeasible_individuals = self.population[~condition]
            top_infeasible_individuals_i = np.argsort(infeasible_individuals["fitness"])[:self.n_elite - len(top_feasible_individuals)]
            top_infeasible_individuals = infeasible_individuals[top_infeasible_individuals_i]

            if self.hybrid:
                self.selection_method(self.population, self.tournament_size, "biased_fitness")
            else:
                self.selection_method(self.population, self.tournament_size, "fitness")
            self.children = np.empty((self.population_size, self.vrp_instance.n_depots + self.vrp_instance.n_customers),
                                     dtype=int)

            self.do_crossover()
            self.do_mutation()

            # Replace old generation with new generation
            self.population["chromosome"] = self.children

            self.do_elitism(top_infeasible_individuals)
            self.do_elitism(top_feasible_individuals)

            # self.fitness_scaling(self.population)
            self.fitness_evaluation()

            if self.hybrid:
                self.diversity_management.calculate_biased_fitness()
                best_ind = self.education_best_individuals()

            # Track number of no improvements
            self.save_fitness_statistics()
            self.save_feasible_stats()

            if self.generation % self.penalty_step == 0:
                self.adjust_penalty()

            if self.hybrid:
                if self.generation % self.survivor_selection_step == 0:
                    self.diversity_management.survivor_selection()
            # elif (self.generation + 1) % self.kill_clone_step == 0:
            #     self.diversity_management.kill_clones()

            if self.hybrid:
                self.do_elitism(np.array([best_ind]))

            best_ind = self.population[np.argsort(self.population["fitness"])[0]]
            if best_ind["fitness"] - 0.0001 < self.fitness_stats[self.generation]["min"]:
                self.fitness_stats[self.generation]["min"] = best_ind["fitness"]
                if (best_ind["capacity_violation"] == 0) and (best_ind["time_warp"] == 0) and (best_ind["duration_violation"]) == 0:
                    self.fitness_stats[self.generation]["min_feasible"] = best_ind["fitness"]

            min_current_fitness = self.fitness_stats[self.generation]["min"]
            if min_current_fitness > self.best_solution["fitness"] - 0.0000001:
                self.no_improvement_counter += 1
                self.diversify_counter += 1
            else:
                self.no_improvement_counter = 0
                self.diversify_counter = 0
                self.best_solution = self.population[np.argmin(self.population["fitness"])].copy()

            # Diversify population
            # if self.diversify_counter >= self.diversify_step:
            #     self.diversity_management.diversity_procedure()

            self.end_time = time.time()
            minutes, seconds = divmod(self.end_time - self.start_time, 60)
            print(
                f"Generation: {self.generation}, n_feasible/infeasible: {self.n_feasible}/{self.population_size - self.n_feasible} Min all/Min feasible/AVG Fitness: "
                f"{self.fitness_stats[self.generation]['min']}/{self.fitness_stats[self.generation]['min_feasible']}/{self.fitness_stats[self.generation]['avg']}, Time: {int(minutes)}:{int(seconds)}")

            # Termination criteria of GA
            if self.end_time - self.start_time >= self.MAX_RUNNING_TIME_IN_S:
                break

    def fitness_evaluation(self):
        for i, chromosome in enumerate(self.population["chromosome"]):
            total_fitness, total_distance, total_capacity_violation, total_time_warp, total_duration_violation = self.decode_chromosome(
                chromosome)
            self.population[i]["fitness"] = total_fitness
            self.population[i]["distance"] = total_distance
            self.population[i]["capacity_violation"] = total_capacity_violation
            self.population[i]["time_warp"] = total_time_warp
            self.population[i]["duration_violation"] = total_duration_violation

    def decode_chromosome(self, chromosome: ndarray, purpose: Purpose = Purpose.FITNESS) -> Tuple[
        ndarray, ndarray, ndarray, ndarray, ndarray]:
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

        selected_values = capacity_complete[np.concatenate([depot_value_index])]
        exceeding_values = selected_values[selected_values > self.vrp_instance.max_capacity]
        differences = exceeding_values - self.vrp_instance.max_capacity
        total_capacity_violation = np.sum(differences)

        selected_values = time_warp_complete[np.concatenate([depot_value_index])]
        total_time_warp = np.sum(selected_values)

        selected_values = duration_complete[np.concatenate([depot_value_index])]
        exceeding_values = selected_values[selected_values > self.vrp_instance.max_duration_of_a_route]
        differences = exceeding_values - self.vrp_instance.max_duration_of_a_route
        total_duration_violation = np.sum(differences)

        zero_indices = np.where(p_complete == 0)[0]
        selected_values = p_complete[np.concatenate([zero_indices - 1])]
        total_fitness = np.sum(selected_values)

        return total_fitness, total_distance, total_capacity_violation, total_time_warp, total_duration_violation

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
            if random() <= self.p_c:
                self.children[individual] = self.crossover.order_beginning(
                    self.crossover.uniform(self.population[individual]["chromosome"],
                                           self.population[individual + 1]["chromosome"]),
                    self.population[individual + 1]["chromosome"])

                self.children[individual + 1] = self.crossover.order_beginning(
                    self.crossover.uniform(self.population[individual + 1]["chromosome"],
                                           self.population[individual]["chromosome"]),
                    self.population[individual]["chromosome"])
            else:
                self.children[individual] = self.population[individual]["chromosome"]
                self.children[individual + 1] = self.population[individual + 1]["chromosome"]

    def do_mutation(self) -> None:
        """
        Handles the mutation
        param: children - empty 1D array
        return: children - 1D array of the mutated children holding chromosome information
        """
        for i, chromosome in enumerate(self.children):
            if random() <= self.p_m:
                if random() < 0.5:
                    for j in range(3):
                        self.mutation.uniform(self.children[i])

                self.mutation.swap(self.children[i])

    def education_best_individuals(self):
        condition = (self.population["capacity_violation"] == 0) & (self.population["time_warp"] == 0) & (
                self.population["duration_violation"] == 0)
        feasible_individuals = self.population[condition]
        if len(feasible_individuals) > 0:
            top_feasible_individual = np.argsort(feasible_individuals["fitness"])[0]
            best_ind = feasible_individuals[top_feasible_individual]
        else:
            infeasible_individuals = self.population[~condition]
            top_infeasible_individual_i = np.argsort(infeasible_individuals["fitness"])[0]
            best_ind = infeasible_individuals[top_infeasible_individual_i]

        diff_fitness = abs(self.education_old_fitness - best_ind["fitness"])
        if diff_fitness > 0.0001:
            new_chromosome, new_fitness = self.education.run(best_ind["chromosome"], best_ind["fitness"])
        else:
            new_chromosome, new_fitness = self.education.run(best_ind["chromosome"], best_ind["fitness"], limited=True)
        total_fitness, total_distance, total_capacity_violation, total_time_warp, total_duration_violation = self.decode_chromosome(new_chromosome)
        old_ind_fitness = best_ind["fitness"]
        self.education_old_fitness = best_ind["fitness"]
        print(best_ind["fitness"], new_fitness, total_fitness)
        if total_fitness - 0.00001 < old_ind_fitness:
            best_ind["chromosome"] = new_chromosome
            best_ind["fitness"] = total_fitness
            best_ind["distance"] = total_distance
            best_ind["capacity_violation"] = total_capacity_violation
            best_ind["time_warp"] = total_time_warp
            best_ind["duration_violation"] = total_duration_violation
        population_indices = list(range(self.population_size))
        shuffle(population_indices)

        # Count educations ran, limit to 2
        counter = 0
        for i in population_indices:
            individual = self.population[i]

            # Same individuals skip
            if individual["fitness"] == best_ind["fitness"] or individual["fitness"] == old_ind_fitness:  # or individual["fitness"] > self.fitness_stats[self.generation]["avg"]:
                continue
            # print(f"RANDOM index: {i},  {individual}")
            counter += 1
            new_chromosome, new_fitness = self.education.run(individual["chromosome"], individual["fitness"])

            total_fitness, total_distance, total_capacity_violation, total_time_warp, total_duration_violation = self.decode_chromosome(new_chromosome)
            print(individual["fitness"], new_fitness, total_fitness)
            if total_fitness - 0.00001 < individual["fitness"] and abs(total_fitness - old_ind_fitness) > 0.00001:
                individual["chromosome"] = new_chromosome
                individual["fitness"] = total_fitness
                individual["distance"] = total_distance
                individual["capacity_violation"] = total_capacity_violation
                individual["time_warp"] = total_time_warp
                individual["duration_violation"] = total_duration_violation

            # print(f"NEW RANDOM index: {i},  {individual}")

            if counter >= 2:
                break

        return best_ind

    def do_elitism(self, top_individuals: ndarray) -> None:
        """
        Perform elitism by replacing the worst individuals with the best individuals
        param: top_individuals - structured 3D array ["individual"]["chromosome]["fitness"]
        """

        worst_individuals_i = np.argsort(self.population["fitness"])[-self.n_elite:]
        # Check if top_individuals is empty or smaller than the number of the worst individuals
        num_to_replace = min(len(top_individuals), len(worst_individuals_i))

        if num_to_replace > 0:
            self.population[worst_individuals_i[:num_to_replace]] = top_individuals[:num_to_replace]

    def adjust_penalty(self):
        # Check conditions for constraint violations
        condition = (self.population["capacity_violation"] == 0) & (
                self.population["time_warp"] == 0) & (
                            self.population["duration_violation"] == 0)

        # Get the indices where the condition is true
        feasible_indices = np.where(condition)[0]
        self.n_feasible = len(feasible_indices)

        if self.n_feasible < self.target_feasible_proportion:
            self.capacity_penalty_factor = self.capacity_penalty_factor * (1 + self.penalty_factor)
            self.duration_penalty_factor = self.duration_penalty_factor * (1 + self.penalty_factor)
            self.time_window_penalty = self.time_window_penalty * (1 + self.penalty_factor)
        else:
            self.capacity_penalty_factor = self.capacity_penalty_factor * (1 - self.penalty_factor)
            self.duration_penalty_factor = self.duration_penalty_factor * (1 - self.penalty_factor)
            self.time_window_penalty = self.time_window_penalty * (1 + self.penalty_factor)

    def print_time_and_text(self, text: str):
        self.end_time = time.time()
        minutes, seconds = divmod(self.end_time - self.start_time, 60)
        print(f"{text}: {int(minutes)}:{int(seconds)}")

    def save_fitness_statistics(self):
        self.fitness_stats[self.generation]["max"] = np.max(self.population["fitness"])
        self.fitness_stats[self.generation]["avg"] = np.mean(self.population["fitness"])
        self.fitness_stats[self.generation]["min"] = np.min(self.population["fitness"])

        condition = (self.population["capacity_violation"] == 0) & (self.population["time_warp"] == 0) & (
                self.population["duration_violation"] == 0)
        feasible_individuals = self.population[condition]
        if len(feasible_individuals) > 0:
            top_feasible_individual_i = np.argsort(feasible_individuals["fitness"])[0]
            top_feasible_individual = feasible_individuals[top_feasible_individual_i]
            self.fitness_stats[self.generation]["min_feasible"] = top_feasible_individual["fitness"]
        else:
            self.fitness_stats[self.generation]["min_feasible"] = 0

    def save_feasible_stats(self):
        self.feasible_stats[self.generation]["feasible"] = self.n_feasible
        self.feasible_stats[self.generation]["infeasible"] = self.population_size - self.n_feasible

    def log_configuration(self, individual, old_best_solution, best_feasible_solution) -> None:
        """
        Logs every relevant parameter
        param: individual - the best solution found
        """

        # Sort population
        sorted_indices = np.argsort(self.population["fitness"])
        self.population[:] = self.population[sorted_indices]

        if not os.path.exists(self.file_prefix_name):
            os.makedirs(self.file_prefix_name)

        with open(os.path.join(self.file_prefix_name, 'best_chromosome.txt'), 'a') as file:
            file.write(f'\nTotal Runtime in seconds: {self.end_time - self.start_time}'
                       f'Parameters:'
                       f'\npopulation_size: {self.population_size}, max_generations: {self.max_generations}, executed generations: {self.generation}, threshold_no_improvement: {self.threshold_no_improvement}'
                       f'\ntarget feasible proportion: {self.target_feasible_proportion}'
                       f'\nfitness_scaling: {self.fitness_scaling.__name__}, selection_method: {self.selection_method.__name__}'
                       f'\np_c: {self.p_c}, p_m: {self.p_m}'
                       f'\ntournament_size: {self.tournament_size}, n_elite: {self.n_elite}'
                       f'\nsurvivor_selection_step: {self.survivor_selection_step}, p_selection_survival: {self.p_selection_survival}'                       
                       f'\nkill clone step: {self.kill_clone_step}'
                       f'\ndiversify_step: {self.diversify_step}, diversify_step: {self.diversify_step}, p_diversify_survival: {self.p_diversify_survival}'
                       f'\nn_closest_neighbors: {self.n_closest_neighbors}, diversity_weight: {self.diversity_weight}, distance_method: {self.distance_method.__name__}'
                       f'\ncapacity_penalty_factor: {self.capacity_penalty_factor}, duration_penalty_factor {self.duration_penalty_factor}, time_window_penalty: {self.time_window_penalty}'
                       f'\nBest fitness all generations, best individual before/after local search: {np.min(self.fitness_stats["min"][self.fitness_stats["min"] != 0])}, {old_best_solution["fitness"]} / {individual["fitness"]}'
                       f'\nBest individual found before local search: {old_best_solution}'
                       f'\nBest individual found after local search: {individual}'
                       f'\nBest feasible individual found: {best_feasible_solution}'
                       f'\n\nFitness stats min: {self.fitness_stats["min"][:self.generation + 1]} '
                       f'\n\nFitness stats feasible min: {self.fitness_stats["min_feasible"][:self.generation + 1]} '
                       f'\n\nFitness stats avg: {self.fitness_stats["avg"][:self.generation + 1]} '
                       f'\n\nFeasible stats: {self.feasible_stats} '
                       f'\nSolution Description: '
                       f'\np: {self.p_complete} '
                       f'\npred: {self.pred_complete} '
                       f'\ndistance {self.distance_complete}'
                       f'\nload: {self.capacity_complete} '
                       f'\ntime: {self.time_complete} '
                       f'\ntime warps: {self.time_warp_complete} '
                       f'\nduration: {self.duration_complete}'
                       f'\n\nAll individuals: {self.population}')
