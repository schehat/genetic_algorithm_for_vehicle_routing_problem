import random

import numpy as np

from vrp_instance import VRPInstance


class Mutation:
    """
    Genetic operator to mutate genetic information of an individual to enhance the diversity of the population
    and local search technique to find better results
    """

    adaptive_mutation_rate = 1  # TODO

    def __init__(self, vrp_instance: VRPInstance, mutation_rate: float):
        self.vrp_instance: VRPInstance = vrp_instance
        self.START_SECOND_PART = vrp_instance.n_depots
        self.START_THIRD_PART = self.START_SECOND_PART + vrp_instance.n_vehicles
        self.END_THIRD_PART = self.START_THIRD_PART + vrp_instance.n_customers
        self.mutation_rate = mutation_rate

    def uniform(self, chromosome: np.ndarray):
        """
        Applying to first and second part of chromosome in place
        param: chromosome 1D array
        """

        # Define the range for mutated numbers of vehicles
        min_vehicle = 0
        max_vehicle = self.vrp_instance.n_vehicles

        # Iterate through the depots and apply uniform mutation
        for i in range(self.START_SECOND_PART):
            if random.random() <= self.mutation_rate:
                mutated_value = np.random.randint(min_vehicle, max_vehicle + 1)
                chromosome[i] = mutated_value

        # Define the range for mutated number of customers
        # Vehicle can have 0 customers to server TODO valid? 
        min_customer = 0
        # max customers limited that every other vehicle can have at least one customer
        max_customer = self.vrp_instance.n_customers  # TODO - self.vrp_instance.n_vehicles + 1

        # Iterate through the vehicles and apply uniform mutation
        for i in range(self.START_SECOND_PART, self.START_THIRD_PART):
            if random.random() <= self.mutation_rate:
                mutated_value = np.random.randint(min_customer, max_customer + 1)
                chromosome[i] = mutated_value

        self._repair_procedure(chromosome)

    def _repair_procedure(self, chromosome: np.ndarray):
        """
        After uniform mutation first and second part of chromosome may break and repair procedure needed
        param: chromosome 1D array
        """

        # iterate through depots
        sum_vehicles = np.sum(chromosome[:self.START_SECOND_PART])
        diff = sum_vehicles - self.vrp_instance.n_vehicles

        while diff != 0:
            random_index = np.random.randint(0, self.START_SECOND_PART)

            if diff > 0:
                # Decrease value if difference positive. But if value = 1 then do nothing
                if chromosome[random_index] > 1:
                    chromosome[random_index] -= 1
                    diff -= 1
            else:
                # Increase value if difference is negative 
                chromosome[random_index] += 1
                diff += 1

        # iterate through vehicles
        sum_customers = np.sum(chromosome[self.START_SECOND_PART:self.START_THIRD_PART])
        diff = sum_customers - self.vrp_instance.n_customers

        while diff != 0:
            random_index = np.random.randint(self.START_SECOND_PART, self.START_THIRD_PART)

            if diff > 0:
                if chromosome[random_index] > 1:
                    chromosome[random_index] -= 1
                    diff -= 1
            else:
                chromosome[random_index] += 1
                diff += 1

    def swap(self, chromosome: np.ndarray):
        """
        Applying to third part of chromosome in place
        param: chromosome 1D array
        """

        # Check if mutation should occur
        if random.random() <= self.mutation_rate:
            positions = self._generate_distinct_positions(2)
            pos1, pos2 = positions[0], positions[1]

            # Perform swap
            chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]

    def inversion(self, chromosome: np.ndarray):
        """
        Applying to third part of chromosome in place
        param: chromosome 1D array
        """

        if random.random() <= self.mutation_rate:
            pos1, pos2 = self._generate_distinct_positions(2)
            pos1, pos2 = min(pos1, pos2), max(pos1, pos2)

            # Perform inversion
            chromosome[pos1:pos2 + 1] = chromosome[pos1:pos2 + 1][::-1]

    # third part of chromosome in place
    def insertion(self, chromosome: np.ndarray):
        """
        Applying to third part of chromosome in place
        param: chromosome 1D array
        """

        if random.random() <= self.mutation_rate:
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
        Utility method to generate 2 distinct integer numbers
        param: num_positions integer number indicating the range
        """

        return np.random.choice(range(self.START_THIRD_PART, self.END_THIRD_PART), size=num_positions, replace=False)
