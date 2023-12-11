from random import random, shuffle
from typing import Tuple

import numpy as np
from numpy import ndarray

from src.utility import set_customer_index_list


class Education:
    """
    Serves as a local search mechanism superior to random mutation
    """

    current_fitness: float
    current_chromosome: ndarray
    new_fitness: float
    customer_index_list: list

    def __init__(self, ga: "GA", max_visit_sequence: int = 3):
        self.ga = ga
        self.max_visit_sequence = max_visit_sequence
        self.neighborhood_iterations = 5

    def run(self, chromosome: ndarray, current_fitness: float) -> Tuple[ndarray, float]:
        """
        Runtime logic
        param: chromosome - being applied education
        return: educated chromosome
        """

        self.current_fitness = current_fitness
        self.current_chromosome = chromosome.copy()
        self.customer_index_list = set_customer_index_list(self.ga.vrp_instance.n_depots, chromosome)

        self.route_improvement(self.current_chromosome.copy())
        self.pattern_improvement()

        return self.current_chromosome, self.current_fitness

    def route_improvement(self, chromosome: ndarray):
        """
        Route improvement deals with the configuration of the customers with the route of a single depot.
        Here is the management of the all the depots held for configuring the new chromosome
        """

        fitness_complete = 0
        chromosome_complete = []

        for depot_i in range(self.ga.vrp_instance.n_depots):
            depot_i, single_chromosome, fitness = self.route_improvement_single_depot(depot_i, chromosome,
                                                                                      self.customer_index_list)
            fitness_complete += fitness
            chromosome_complete.append((depot_i, single_chromosome))

        # chromosome_complete = sorted(chromosome_complete, key=lambda x: x[0])
        full_chromosome = []
        # Append depot_info in one block and then flattened customer_info
        full_chromosome.extend(chromosome[:self.ga.vrp_instance.n_depots])
        full_chromosome.extend(customer for _, customer_info in chromosome_complete for customer in customer_info)

        if fitness_complete < self.current_fitness:
            self.current_fitness = fitness_complete
            self.current_chromosome = np.array(full_chromosome)

    def route_improvement_single_depot(self, depot_i, chromosome, customer_index_list) -> Tuple[int, list, float]:
        """
        Responsible for running route improvement for a single depot
        param: depot_i - depot index to identify depot
        return: depot_i, new single depot chromosome WITHOUT depot_information as first element, new fitness
        """

        depot_i_n_customers = chromosome[depot_i]
        # Contains customer chromosome for one depot
        single_depot_chromosome = list(
            chromosome[customer_index_list[depot_i]: customer_index_list[depot_i] + depot_i_n_customers])
        # Add depot information at the beginning of chromosome for the split algorithm
        single_depot_chromosome.insert(0, depot_i_n_customers)

        # Make a copy to keep the original list intact only with the customers
        shuffle_single_depot_chromosome = single_depot_chromosome[1:]
        # Shuffle to achieve random order which customer is selected
        shuffle(shuffle_single_depot_chromosome)

        best_fitness = float("inf")
        best_insert_position = None
        # max_improvements = 2
        # max_depot_iterations = 5
        for customer in shuffle_single_depot_chromosome:
            temp = single_depot_chromosome.copy()
            best_fitness = self.ga.split.split_single_depot(single_depot_chromosome, 0, 1)[0][-1]
            single_depot_chromosome = [x for i, x in enumerate(single_depot_chromosome) if x != customer or i == 0]

            # n_improvements = 0
            # Starting from 1 to exclude depot and until len + 1 to add as last element
            shuffle_insertion = list(range(1, len(single_depot_chromosome) + 1))
            shuffle(shuffle_insertion)
            for insert_position in shuffle_insertion:  # [:max_depot_iterations]:
                # Insert the customer at the specified position
                single_depot_chromosome.insert(insert_position, customer)

                # Call split to calculate fitness and only get the total fitness return value. Last two arguments special cases
                # using single depot split depot_i=0 and customer_offset=1 because single chromosome passed
                fitness = self.ga.split.split_single_depot(single_depot_chromosome, 0, 1)[0][-1]

                # Update the best fitness and position if needed
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_insert_position = insert_position
                    # n_improvements += 1

                # Remove the customer for the next iteration
                single_depot_chromosome.pop(insert_position)
                # if n_improvements >= max_improvements:
                #     break

            if best_insert_position is not None:
                single_depot_chromosome.insert(best_insert_position, customer)
            else:
                single_depot_chromosome = temp.copy()

        # Remove depot information at the end
        return depot_i, single_depot_chromosome[1:], best_fitness

    def pattern_improvement(self):
        """
        Pattern Improvement deals with the configuration of all customers between different depots to
        evaluate better depot assignment
        """
        # Define the three methods
        methods = [self.n1_swap_and_relocate, self.n2_2opt_asterisk, self.n3_2opt]
        # Shuffle the list of methods randomly
        shuffle(methods)
        # Execute the methods in the shuffled order
        fitness = self.current_fitness
        chromosome = self.current_chromosome.copy()
        for method in methods:
            # Sets the best candidate as the new self.chromosome
            chromosome, fitness = self.run_neighborhood_search(method, chromosome, fitness)

        if fitness < self.current_fitness:
            self.current_fitness = fitness
            self.current_chromosome = chromosome

    def run_neighborhood_search(self, neighborhood_search, chromosome: ndarray, fitness) -> Tuple[ndarray, float]:
        """
        Runs each neighbor method according to the neighbor exploration of 5% and picks the best solution
        """

        best_chromosome = chromosome
        best_fitness = fitness

        # Run neighborhood search
        for _ in range(self.neighborhood_iterations):
            chromosome_candidate = neighborhood_search(chromosome, self.customer_index_list)
            fitness_candidate = self.ga.decode_chromosome(chromosome_candidate)[0]

            # Update the best candidate and best fitness if needed
            if fitness_candidate < best_fitness:
                best_chromosome = chromosome_candidate
                best_fitness = fitness_candidate

        return best_chromosome, best_fitness

    def n1_swap_and_relocate(self, chromosome, customer_index_list) -> ndarray:
        """
        Swaps and inverses if necessary genes between the same or different depots
        return: new chromosome
        """

        depot1, depot2 = self._pick_random_depots(False)

        # Select visit sequence length to swap
        seq_length_depot1 = np.random.randint(0, self.max_visit_sequence + 1)
        seq_length_depot2 = np.random.randint(0, self.max_visit_sequence + 1)

        # Skip if no manipulation
        if seq_length_depot1 + seq_length_depot2 == 0:
            return chromosome

        # Determine the start and end indices for the chosen depots
        start_depot1, end_depot1 = self._determine_start_and_end_depot_position(depot1, chromosome, customer_index_list)
        start_depot2, end_depot2 = self._determine_start_and_end_depot_position(depot2, chromosome, customer_index_list)

        # Ensure the selected sequence length is valid for the depots
        seq_length_depot1 = min(seq_length_depot1, end_depot1 - start_depot1 + 1)
        seq_length_depot2 = min(seq_length_depot2, end_depot2 - start_depot2 + 1)

        # Pick random starting point in block. +1 for extra element and +1 because exclusive high value
        start_pick_depot1 = start_depot1 + np.random.randint(0, end_depot1 - start_depot1 + 1 - seq_length_depot1 + 1)
        start_pick_depot2 = start_depot2 + np.random.randint(0, end_depot2 - start_depot2 + 1 - seq_length_depot2 + 1)

        # Edge case if same depot picked and ranges overlap. TODO: fix
        # if depot1 == depot2:
        #     # Check if the ranges from depot1 and depot2 are overlapping
        #     if start_pick_depot1 <= start_pick_depot2 < start_pick_depot1 + seq_length_depot1 \
        #             or start_pick_depot1 <= start_pick_depot2 + seq_length_depot2 < start_pick_depot1 + seq_length_depot1:
        #         # Check if enough genes exist for both sequences
        #         if end_depot1 - start_depot1 + 1 > seq_length_depot1 + seq_length_depot2:
        #             # Iterate until genes do not overlap
        #             while start_pick_depot1 <= start_pick_depot2 < start_pick_depot1 + seq_length_depot1 \
        #                     or start_pick_depot1 <= start_pick_depot2 + seq_length_depot2 < start_pick_depot1 + seq_length_depot1:
        #                 start_pick_depot1 = start_depot1 + np.random.randint(0,
        #                                                                      end_depot1 - start_depot1 + 1 - seq_length_depot1 + 1)
        #                 start_pick_depot2 = start_depot2 + np.random.randint(0,
        #                                                                      end_depot2 - start_depot2 + 1 - seq_length_depot2 + 1)
        #         else:
        #             # Assign static if len of both sequences = len of block, because random assigning unnecessary
        #             start_pick_depot1 = start_depot1
        #             start_pick_depot2 = start_pick_depot1 + seq_length_depot1

        # Extract the selected genes. No need for +1, last value already included
        swapping_genes1 = chromosome[start_pick_depot1: start_pick_depot1 + seq_length_depot1]
        swapping_genes2 = chromosome[start_pick_depot2: start_pick_depot2 + seq_length_depot2]

        # Invert the genes
        swapping_genes1 = swapping_genes1[::-1] if np.random.rand() < 0.5 else swapping_genes1
        swapping_genes2 = swapping_genes2[::-1] if np.random.rand() < 0.5 else swapping_genes2

        # Adjust depot information
        chromosome[depot1] += len(swapping_genes2) - len(swapping_genes1)
        chromosome[depot2] += len(swapping_genes1) - len(swapping_genes2)

        # Swap the genes in the chromosome
        return np.concatenate([
            chromosome[:start_pick_depot1],
            swapping_genes2,
            chromosome[start_pick_depot1 + seq_length_depot1: start_pick_depot2],
            swapping_genes1,
            chromosome[
            start_pick_depot2 + seq_length_depot2:self.ga.vrp_instance.n_depots + self.ga.vrp_instance.n_customers]
        ])

    def n2_2opt_asterisk(self, chromosome, customer_index_list) -> ndarray:
        """
        Swaps customers at the extremities of distinct routes
        return: new chromosome
        """

        depot1, depot2 = self._pick_random_depots(False)

        # Select visit sequence length to swap
        seq_length_depot1 = np.random.randint(0, self.max_visit_sequence + 1)
        seq_length_depot2 = np.random.randint(0, self.max_visit_sequence + 1)

        # Skip if no manipulation
        if seq_length_depot1 + seq_length_depot2 == 0:
            return chromosome

        # Determine the start and end indices for the chosen depots
        start_depot1, end_depot1 = self._determine_start_and_end_depot_position(depot1, chromosome, customer_index_list)
        start_depot2, end_depot2 = self._determine_start_and_end_depot_position(depot2, chromosome, customer_index_list)

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
        swapping_genes1 = chromosome[start_pick_depot1: start_pick_depot1 + seq_length_depot1]
        swapping_genes2 = chromosome[start_pick_depot2: start_pick_depot2 + seq_length_depot2]

        # Adjust depot information
        chromosome[depot1] += len(swapping_genes2) - len(swapping_genes1)
        chromosome[depot2] += len(swapping_genes1) - len(swapping_genes2)

        # Swap the genes in the chromosome
        return np.concatenate([
            chromosome[:start_pick_depot1],
            swapping_genes2,
            chromosome[start_pick_depot1 + seq_length_depot1: start_pick_depot2],
            swapping_genes1,
            chromosome[start_pick_depot2 + seq_length_depot2:]
        ])

    def n3_2opt(self, chromosome, customer_index_list) -> ndarray:
        """
        Inverse customer sequence, between different depots possible
        return: new chromosome
        """

        # Select a sequence length to inverse
        seq_length = np.random.randint(1, self.max_visit_sequence + 1)

        # Skip if no manipulation
        if seq_length == 1:
            return chromosome

        # Determine the start and end indices for the chosen sequence
        start_index = np.random.randint(customer_index_list[0], len(chromosome) - seq_length)

        # Extract the selected sequence
        selected_sequence = chromosome[start_index:start_index + seq_length]

        # Reverse the sequence
        reversed_sequence = selected_sequence[::-1]

        # Swap the reversed sequence in the chromosome
        new_chromosome = np.copy(chromosome)
        new_chromosome[start_index:start_index + seq_length] = reversed_sequence

        return new_chromosome

    def _pick_random_depots(self, replace: bool) -> Tuple[int, int]:
        # Select two depots randomly and ensure depot_1 < depot_2
        depot1, depot2 = np.random.choice(self.ga.vrp_instance.n_depots, size=2, replace=replace)
        return min(depot1, depot2), max(depot1, depot2)

    def _determine_start_and_end_depot_position(self, depot_i: int, chromosome: ndarray, customer_index_list: list) -> \
    Tuple[int, int]:
        start_depot = customer_index_list[depot_i]
        if depot_i == self.ga.vrp_instance.n_depots - 1:
            end_depot = len(chromosome) - 1
        else:
            end_depot = customer_index_list[depot_i + 1] - 1
        return start_depot, end_depot
