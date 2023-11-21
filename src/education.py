from concurrent.futures import ThreadPoolExecutor
from random import random, shuffle
from typing import Tuple

import numpy as np
from numpy import ndarray


class Education:
    chromosome = None
    customer_index_list = None

    def __init__(self, ga: "GA"):
        self.ga = ga

    def run(self, chromosome: ndarray) -> None:
        self.chromosome = chromosome
        self.route_improvement()
        self.pattern_improvement()
        self.route_improvement()

    def route_improvement(self) -> None:
        # Determine indices for chromosome "splitting"
        customer_index = self.ga.vrp_instance.n_depots
        self.customer_index_list = [customer_index]
        for depot_i in range(self.ga.vrp_instance.n_depots - 1):
            customer_index += self.chromosome[depot_i]
            self.customer_index_list.append(customer_index)

        fitness_complete = 0
        chromosome_complete = []

        # Parallel execution
        with ThreadPoolExecutor() as executor:
            results = [executor.submit(self.route_improvement_single_depot, depot_i) for depot_i in
                       range(self.ga.vrp_instance.n_depots)]
            for future in results:
                fitness, chromosome = future.result()
                fitness_complete += fitness
                chromosome_complete += chromosome

        self.ga.total_fitness = fitness

        # TODO: Normal split to log correctly?

    def route_improvement_single_depot(self, depot_i) -> Tuple[float, list]:
        depot_i_n_customers = self.chromosome[depot_i]
        # Contains customer chromosome for one depot
        single_depot_chromosome = list(
            self.chromosome[self.customer_index_list[depot_i]: self.customer_index_list[depot_i] + depot_i_n_customers])
        # Add depot information at the beginning of chromosome for the split algorithm
        single_depot_chromosome.insert(0, depot_i_n_customers)

        # Make a copy to keep the original list intact only with the customers
        shuffle_single_depot_chromosome = single_depot_chromosome[1:]
        # Shuffle to achieve random order which customer is selected
        shuffle(shuffle_single_depot_chromosome)

        best_fitness = float('inf')
        best_insert_position = None
        for customer in shuffle_single_depot_chromosome:
            single_depot_chromosome.remove(customer)
            # Depot information might be removed so need to insert it back
            if single_depot_chromosome[0] != depot_i_n_customers:
                single_depot_chromosome.insert(0, depot_i_n_customers)

            # Starting from 1 to exclude depot and until len + 1 to add as last element
            for insert_position in range(1, len(single_depot_chromosome) + 1):
                # Insert the customer at the specified position
                single_depot_chromosome.insert(insert_position, customer)

                # Call split to calculate fitness and only get the total fitness return value.
                # Last argument customer_offset always 1
                fitness = self.ga.split.split_single_depot(single_depot_chromosome, 0, 1)[0][-1]

                # Update the best fitness and position if needed
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_insert_position = insert_position

                # Remove the customer for the next iteration
                single_depot_chromosome.remove(customer)
                # Depot information might be removed so need to insert it back
                if single_depot_chromosome[0] != depot_i_n_customers:
                    single_depot_chromosome.insert(0, depot_i_n_customers)

            single_depot_chromosome.insert(best_insert_position, customer)

        return best_fitness, single_depot_chromosome

    def pattern_improvement(self):
        pass
