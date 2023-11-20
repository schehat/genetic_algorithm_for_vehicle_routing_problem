from concurrent.futures import ThreadPoolExecutor
from random import random

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

        # Parallel execution
        with ThreadPoolExecutor() as executor:
            results = [executor.submit(self.route_improvement_single_depot, depot_i) for depot_i in range(self.ga.vrp_instance.n_depots)]
            for future in results:
                fitness = future.result()
                fitness_complete += fitness

        self.ga.total_fitness = fitness

        # TODO: Normal split to log correctly?

    def route_improvement_single_depot(self, depot_i) -> float:
        depot_i_n_customers = self.chromosome[depot_i]
        customer_candidates = list(range(self.customer_index_list[depot_i], self.customer_index_list[depot_i] + depot_i_n_customers))

        # Shuffle customer indices to visit them in random order
        random.shuffle(customer_candidates)

        best_fitness = float('inf')
        best_insert_position = None
        # Make a copy to keep the original list intact
        temp_customer_candidates = customer_candidates.copy()

        for customer_index in customer_candidates:
            temp_customer_candidates.remove(customer_index)

            for insert_position in range(len(temp_customer_candidates) + 1):
                # Insert the customer at the specified position
                temp_customer_candidates.insert(insert_position, customer_index)

                # Call split to calculate fitness
                self.split_single_depot(self.chromosome, depot_i, temp_customer_candidates)

                # Update best fitness and position if needed
                if self.ga.total_fitness < best_fitness:
                    best_fitness = self.ga.total_fitness
                    best_insert_position = insert_position

                # Remove the customer for the next iteration
                temp_customer_candidates.remove(customer_index)

        # Final insertion at the best position
        temp_customer_candidates.insert(best_insert_position, customer_index)

        return best_fitness



    def pattern_improvement(self):
        pass
