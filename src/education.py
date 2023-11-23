import math
from concurrent.futures import ThreadPoolExecutor
from random import random, shuffle, seed
from typing import Tuple

import numpy as np
from numpy import ndarray


class Education:
    chromosome = None
    customer_index_list = None

    def __init__(self, ga: "GA"):
        self.ga = ga
        self.neighborhood_iterations = math.ceil(0.05 * (ga.vrp_instance.n_depots + ga.vrp_instance.n_customers))

    def run(self, chromosome: ndarray) -> ndarray:
        self.chromosome = chromosome
        customer_index = self.ga.vrp_instance.n_depots
        self.customer_index_list = [customer_index]
        for depot_i in range(self.ga.vrp_instance.n_depots - 1):
            customer_index += self.chromosome[depot_i]
            self.customer_index_list.append(customer_index)

        # self.route_improvement()
        self.pattern_improvement()
        # self.route_improvement()

        return self.chromosome

    def route_improvement(self) -> None:
        # Determine indices for chromosome "splitting"
        customer_index = self.ga.vrp_instance.n_depots
        self.customer_index_list = [customer_index]
        for depot_i in range(self.ga.vrp_instance.n_depots - 1):
            customer_index += self.chromosome[depot_i]
            self.customer_index_list.append(customer_index)

        fitness_complete = 0
        chromosome_complete = []

        for depot_i in range(self.ga.vrp_instance.n_depots):
            depot_i, single_chromosome, fitness = self.route_improvement_single_depot(depot_i)
            fitness_complete += fitness
            chromosome_complete.append((depot_i, single_chromosome))

        # # Parallel execution
        # with ThreadPoolExecutor() as executor:
        #     results = [executor.submit(self.route_improvement_single_depot, depot_i) for depot_i in
        #                range(self.ga.vrp_instance.n_depots)]
        #     for future in results:
        #         depot_i, single_chromosome, fitness = future.result()
        #         fitness_complete += fitness
        #         chromosome_complete.append((depot_i, single_chromosome))

        chromosome_complete = sorted(chromosome_complete, key=lambda x: x[0])
        full_chromosome = []
        # Append depot_info in one block and then flattened customer_info
        full_chromosome.extend(self.chromosome[:self.ga.vrp_instance.n_depots])
        full_chromosome.extend(customer for _, customer_info in chromosome_complete for customer in customer_info)

        self.chromosome = np.array(full_chromosome)
        self.ga.total_fitness = fitness_complete

    def route_improvement_single_depot(self, depot_i) -> Tuple[int, list, float]:
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
            single_depot_chromosome = [x for i, x in enumerate(single_depot_chromosome) if x != customer or i == 0]

            # Starting from 1 to exclude depot and until len + 1 to add as last element
            for insert_position in range(1, len(single_depot_chromosome) + 1):
                # Insert the customer at the specified position
                single_depot_chromosome.insert(insert_position, customer)

                # Call split to calculate fitness and only get the total fitness return value. Last two arguments special cases
                # using single depot split depot_i=0 and customer_offset=1 because single chromosome passed
                fitness = self.ga.split.split_single_depot(single_depot_chromosome, 0, 1)[0][-1]

                # Update the best fitness and position if needed
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_insert_position = insert_position

                # Remove the customer for the next iteration
                single_depot_chromosome.pop(insert_position)

            single_depot_chromosome.insert(best_insert_position, customer)

        # Remove depot information at the end
        return depot_i, single_depot_chromosome[1:], best_fitness

    def pattern_improvement(self):
        # best_candidate = None
        # best_fitness = float('inf')
        #
        # # Run neighborhood search
        # for _ in range(self.neighborhood_iterations):
        #     chromosome_candidate = self.n1_swap_and_relocate()
        #     self.ga.split.split(chromosome_candidate)
        #     zero_indices = np.where(self.ga.p_complete == 0)[0]
        #     selected_values = self.ga.p_complete[np.concatenate([zero_indices - 1])]
        #     chromosome_candidate_fitness = np.sum(selected_values)
        #
        #     # Update the best candidate and best fitness if needed
        #     if chromosome_candidate_fitness < best_fitness:
        #         best_candidate = chromosome_candidate
        #         best_fitness = chromosome_candidate_fitness
        #
        # # Set the best candidate as the new chromosome
        # self.chromosome = best_candidate

        chromosome_candidate = self.n2_2opt_asterisk()
        # self.n3_2opt()

    def n1_swap_and_relocate(self) -> ndarray:
        depot1, depot2 = self.pick_random_depots(True)

        # Select visit sequence length to swap
        seq_length_depot1 = np.random.randint(0, 3)
        seq_length_depot2 = np.random.randint(0, 3)

        # Determine the start and end indices for the chosen depots
        start_depot1, end_depot1 = self.determine_start_and_end_depot_position(depot1)
        start_depot2, end_depot2 = self.determine_start_and_end_depot_position(depot2)

        # Ensure the selected sequence length is valid for the depots
        seq_length_depot1 = min(seq_length_depot1, end_depot1 - start_depot1 + 1)
        seq_length_depot2 = min(seq_length_depot2, end_depot2 - start_depot2 + 1)

        # Pick random starting point in block. +1 for extra element and +1 because exclusive high value
        start_pick_depot1 = start_depot1 + np.random.randint(0, end_depot1 - start_depot1 + 1 - seq_length_depot1 + 1)
        start_pick_depot2 = start_depot2 + np.random.randint(0, end_depot2 - start_depot2 + 1 - seq_length_depot2 + 1)

        # Edge case if same depot picked and ranges overlap
        if depot1 == depot2:
            # Check if the ranges from depot1 and depot2 are overlapping
            if start_pick_depot1 <= start_pick_depot2 < start_pick_depot1 + seq_length_depot1 \
                    or start_pick_depot1 <= start_pick_depot2 + seq_length_depot2 < start_pick_depot1 + seq_length_depot1:
                # Check if enough genes exist for both sequences
                if end_depot1 - start_depot1 + 1 > seq_length_depot1 + seq_length_depot2:
                    # Iterate until genes do not overlap
                    while start_pick_depot1 <= start_pick_depot2 < start_pick_depot1 + seq_length_depot1 \
                            or start_pick_depot1 <= start_pick_depot2 + seq_length_depot2 < start_pick_depot1 + seq_length_depot1:
                        start_pick_depot1 = start_depot1 + np.random.randint(0,
                                                                             end_depot1 - start_depot1 + 1 - seq_length_depot1 + 1)
                        start_pick_depot2 = start_depot2 + np.random.randint(0,
                                                                             end_depot2 - start_depot2 + 1 - seq_length_depot2 + 1)
                else:
                    # Assign static if len of both sequences = len of block, because random assigning unnecessary
                    start_pick_depot1 = start_depot1
                    start_pick_depot2 = start_pick_depot1 + seq_length_depot1

        # Extract the selected genes. No need for +1, last value already included
        swapping_genes1 = self.chromosome[start_pick_depot1: start_pick_depot1 + seq_length_depot1]
        swapping_genes2 = self.chromosome[start_pick_depot2: start_pick_depot2 + seq_length_depot2]

        # Invert the genes
        swapping_genes1 = swapping_genes1[::-1] if np.random.rand() < 0.5 else swapping_genes1
        swapping_genes2 = swapping_genes2[::-1] if np.random.rand() < 0.5 else swapping_genes2

        # Adjust depot information
        self.chromosome[depot1] += len(swapping_genes2) - len(swapping_genes1)
        self.chromosome[depot2] += len(swapping_genes1) - len(swapping_genes2)

        # Swap the genes in the chromosome
        return np.concatenate([
            self.chromosome[:start_pick_depot1],
            swapping_genes2,
            self.chromosome[start_pick_depot1 + seq_length_depot1: start_pick_depot2],
            swapping_genes1,
            self.chromosome[start_pick_depot2 + seq_length_depot2:]
        ])

    def n2_2opt_asterisk(self):
        depot1, depot2 = self.pick_random_depots(False)

        # Select visit sequence length to swap
        seq_length_depot1 = np.random.randint(0, 3)
        seq_length_depot2 = np.random.randint(0, 3)

        # Determine the start and end indices for the chosen depots
        start_depot1, end_depot1 = self.determine_start_and_end_depot_position(depot1)
        start_depot2, end_depot2 = self.determine_start_and_end_depot_position(depot2)

        # Ensure the selected sequence length is valid for the depots
        seq_length_depot1 = min(seq_length_depot1, end_depot1 - start_depot1 + 1)
        seq_length_depot2 = min(seq_length_depot2, end_depot2 - start_depot2 + 1)

        # Pick random starting point at the start of block or end of block
        if random() < 0.5:
            start_pick_depot1 = start_depot1
        else:
            start_pick_depot1 = end_depot1 - seq_length_depot1 + 1
        if random() < 0.5:
            start_pick_depot2 = start_depot2
        else:
            start_pick_depot2 = end_depot2 - seq_length_depot2 + 1

        # Extract the selected genes. No need for +1, last value already included
        swapping_genes1 = self.chromosome[start_pick_depot1: start_pick_depot1 + seq_length_depot1]
        swapping_genes2 = self.chromosome[start_pick_depot2: start_pick_depot2 + seq_length_depot2]

        # Adjust depot information
        self.chromosome[depot1] += len(swapping_genes2) - len(swapping_genes1)
        self.chromosome[depot2] += len(swapping_genes1) - len(swapping_genes2)

        # Swap the genes in the chromosome
        return np.concatenate([
            self.chromosome[:start_pick_depot1],
            swapping_genes2,
            self.chromosome[start_pick_depot1 + seq_length_depot1: start_pick_depot2],
            swapping_genes1,
            self.chromosome[start_pick_depot2 + seq_length_depot2:]
        ])

    def n3_2opt(self):
        pass

    def pick_random_depots(self, replace: bool) -> Tuple[int, int]:
        # Select two depots randomly and ensure depot_1 < depot_2
        depot1, depot2 = np.random.choice(self.ga.vrp_instance.n_depots, size=2, replace=replace)
        return min(depot1, depot2), max(depot1, depot2)

    def determine_start_and_end_depot_position(self, depot_i: int) -> Tuple[int, int]:
        start_depot = self.customer_index_list[depot_i]
        if depot_i == self.ga.vrp_instance.n_depots - 1:
            end_depot = len(self.chromosome) - 1
        else:
            end_depot = self.customer_index_list[depot_i + 1] - 1
        return start_depot, end_depot
