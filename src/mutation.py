from random import random
from math import floor, ceil

import numpy as np

from vrp import VRPInstance


class Mutation:
    """
    Genetic operator to mutate genetic information of an individual to enhance the diversity of the population
    and local search technique to find better results
    """

    P_UNIFORM = 0.5

    def __init__(self, vrp_instance: VRPInstance, minimum_serving_customers_per_vehicle=0.2):
        self.vrp_instance: VRPInstance = vrp_instance
        # Helper attributes
        self.END_SECOND_PART = self.vrp_instance.n_depots + vrp_instance.n_customers
        self.minimum_serving_customers_per_vehicle = minimum_serving_customers_per_vehicle

        # MIN_CUSTOMER = floor((self.vrp_instance.n_customers / self.vrp_instance.n_depots) * 0.5)
        # MAX_CUSTOMER = ceil((self.vrp_instance.n_customers / self.vrp_instance.n_depots) * 2.0)

    def uniform(self, chromosome: np.ndarray):
        """
        Applying uniform mutation to first part of chromosome in place
        param: chromosome 1D array
        """

        # Define the range for mutated number of customers. TODO: is ok?
        # min_customer = floor((self.vrp_instance.n_customers / self.vrp_instance.n_depots) * 0.5)
        # max_customer = ceil((self.vrp_instance.n_customers / self.vrp_instance.n_depots) * 2.0)
        #
        # # Iterate through the vehicles and apply uniform mutation
        # for i in range(self.vrp_instance.n_depots):
        #     if random() <= self.P_UNIFORM:
        #         mutated_value = np.random.randint(min_customer, max_customer + 1)
        #         chromosome[i] = mutated_value
        #
        # self._repair_procedure(chromosome)

        depot1, depot2 = np.random.choice(self.vrp_instance.n_depots, size=2, replace=False)
        chromosome[depot1] = chromosome[depot1] + 1
        chromosome[depot2] = chromosome[depot2] - 1

    def _repair_procedure(self, chromosome: np.ndarray):
        """
        After uniform mutation first part of chromosome may break and repair procedure needed
        param: chromosome - 1D array
        """

        # iterate through depots
        sum_customers = np.sum(chromosome[:self.vrp_instance.n_depots])
        diff = sum_customers - self.vrp_instance.n_customers

        while diff != 0:
            random_index = np.random.randint(self.vrp_instance.n_depots)

            if diff > 0:
                if chromosome[random_index] > 1:
                    chromosome[random_index] -= 1
                    diff -= 1
            else:
                chromosome[random_index] += 1
                diff += 1

    def swap(self, chromosome: np.ndarray):
        """
        Applying swap mutation to second part of chromosome in place
        param: chromosome - 1D array
        """

        positions = self._generate_distinct_positions(2)
        pos1, pos2 = positions[0], positions[1]

        # Perform swap
        chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]

    def inversion(self, chromosome: np.ndarray):
        """
        Applying inversion mutation to second part of chromosome in place
        param: chromosome - 1D array
        """

        pos1, pos2 = self._generate_distinct_positions(2)
        pos1, pos2 = min(pos1, pos2), max(pos1, pos2)

        # Perform inversion
        chromosome[pos1:pos2 + 1] = chromosome[pos1:pos2 + 1][::-1]

    def insertion(self, chromosome: np.ndarray):
        """
        Applying insertion mutation to second part of chromosome in place
        param: chromosome - 1D array
        """

        pos1, pos2 = self._generate_distinct_positions(2)
        pos1, pos2 = min(pos1, pos2), max(pos1, pos2)

        # Get the customer to insert
        gene_to_insert = chromosome[pos2]

        # Remove the customer from its original position
        removed_gene = np.delete(chromosome, pos2)

        # Insert the customer after pos1
        chromosome[:] = np.insert(removed_gene, pos1 + 1, gene_to_insert)

    def _generate_distinct_positions(self, num_positions):
        """
        Generate random distinct integer numbers
        param: num_positions integer number indicating the range
        """

        return np.random.choice(range(self.vrp_instance.n_depots, self.END_SECOND_PART), size=num_positions, replace=False)
