import random

import numpy as np

from vrp_instance import VRPInstance


class Mutation():
    adaptive_mutation_rate = 1 #TODO

    def __init__(self, vrp_instance: VRPInstance, mutation_rate: float):
        self.vrp_instance = vrp_instance
        self.START_SECOND_PART = vrp_instance.n_vehicles
        self.END_SECOND_PART = self.START_SECOND_PART + vrp_instance.n_customers
        self.mutation_rate = mutation_rate

    # first part of chromosome in place
    def uniform(self, chromosome: np.ndarray):
        # Define the range for mutated number of customers
        # Every vehicle at least 1 customer 
        min_customer = 1
        # max customers limited that every other vehicle can have at least one customer
        max_customer = self.vrp_instance.n_customers - self.vrp_instance.n_vehicles + 1

        # Iterate through the vehicle genes and apply uniform mutation
        for i in range(self.vrp_instance.n_vehicles):
            if random.random() <= self.mutation_rate:
                mutated_value = np.random.randint(min_customer, max_customer + 1)
                chromosome[i] = mutated_value

        self.repair_procedure(chromosome)

    # After uniform mutation first part of chromosome may break and repair procedure needed 
    def repair_procedure(self, chromosome: np.ndarray):
        sum_customers = np.sum(chromosome[:self.vrp_instance.n_vehicles])
        diff = sum_customers - self.vrp_instance.n_customers 

        while diff != 0:
            random_index = np.random.randint(0, self.vrp_instance.n_vehicles)

            if diff > 0:
                # Decrease value if difference positive. But if value = 1 then do nothing
                if chromosome[random_index] > 1:
                    chromosome[random_index] -= 1
                    diff -= 1
            else:
                # Increase value if difference is negative 
                chromosome[random_index] += 1
                diff += 1

    # second part of chromosome in place
    def swap(self, chromosome: np.ndarray):
        # Check if mutation should occur
        if random.random() <= self.mutation_rate:            
            # Ensure two distinct random positions
            positions = np.random.choice(range(self.START_SECOND_PART, self.END_SECOND_PART), size=2, replace=False)
            pos1, pos2 = positions[0], positions[1]

            # Perform swap in place
            chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]

    # second part of chromosome in place
    def inversion(self, chromosome: np.ndarray):
       # Check if mutation should occur
        if random.random() <= self.mutation_rate:
            # Ensure two distinct random positions
            pos1, pos2 = np.random.choice(range(self.START_SECOND_PART, self.END_SECOND_PART), size=2, replace=False)
            # Ensure pos1 < pos2
            pos1, pos2 = min(pos1, pos2), max(pos1, pos2)

            # Perform inversion in place
            chromosome[pos1:pos2 + 1] = chromosome[pos1:pos2 + 1][::-1]

    # second part of chromosome in place
    def insertion(self, chromosome: np.ndarray):
        # Check if mutation should occur
        if random.random() <= self.mutation_rate:
            # Ensure two distinct random positions
            pos1, pos2 = np.random.choice(range(self.START_SECOND_PART, self.END_SECOND_PART), size=2, replace=False)
            # Ensure pos1 < pos2
            pos1, pos2 = min(pos1, pos2), max(pos1, pos2)

            # Get the customer to insert
            gene_to_insert = chromosome[pos2]

            # Remove the customer from its original position
            removed_gene = np.delete(chromosome, pos2)

            # Insert the customer after pos1
            chromosome[:] = np.insert(removed_gene, pos1 + 1, gene_to_insert)
