import random

import numpy as np

from chromosome import Chromosome


class Mutation():
    adaptive_mutation_rate = 1 #TODO

    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    # Mutation first part of chromosome
    def uniform(self, chromosome: Chromosome):
        # Define the range for mutated values
        # Every vehicle at least 1 customer 
        min_customer = 1
        # max customers limited that every other vehicle can have at least one customer
        max_customer = Chromosome.N_CUSTOMERS - Chromosome.N_DEPOTS * Chromosome.N_VEHICLES_PER_DEPOT + 1

        # Iterate through the genes and apply uniform mutation
        for i in range(Chromosome.N_DEPOTS * Chromosome.N_VEHICLES_PER_DEPOT):
            if random.random() <= self.mutation_rate:
                mutated_value = np.random.randint(min_customer, max_customer + 1)
                chromosome.genes[i] = mutated_value

    # Mutation second part of chromosome
    def swap(self, chromosome: Chromosome):
        # Check if mutation should occur
        if random.random() <= self.mutation_rate:
            start = Chromosome.N_DEPOTS * Chromosome.N_VEHICLES_PER_DEPOT
            end = start + Chromosome.N_CUSTOMERS
            
            # Ensure two distinct random positions
            positions = np.random.choice(range(start, end), size=2, replace=False)
            pos1, pos2 = positions[0], positions[1]

            # Perform swap in place
            chromosome.genes[pos1], chromosome.genes[pos2] = chromosome.genes[pos2], chromosome.genes[pos1]

    # Mutation second part of chromosome
    def inversion(self, chromosome: Chromosome):
       # Check if mutation should occur
        if random.random() <= self.mutation_rate:
            start = Chromosome.N_DEPOTS * Chromosome.N_VEHICLES_PER_DEPOT
            end = start + Chromosome.N_CUSTOMERS

            # Ensure two distinct random positions
            pos1, pos2 = np.random.choice(range(start, end), size=2, replace=False)
            # Ensure pos1 < pos2
            pos1, pos2 = min(pos1, pos2), max(pos1, pos2)

            # Perform inversion in place
            chromosome.genes[pos1:pos2 + 1] = genes[pos1:pos2 + 1][::-1]

    # Mutation second part of chromosome
    def insertion(self, chromosome: Chromosome):
         # Check if mutation should occur
        if random.random() <= self.mutation_rate:
            start = Chromosome.N_DEPOTS * Chromosome.N_VEHICLES_PER_DEPOT
            end = start + Chromosome.N_CUSTOMERS

            # Ensure two distinct random positions
            pos1, pos2 = np.random.choice(range(start, end), size=2, replace=False)
            # Ensure pos1 < pos2
            pos1, pos2 = min(pos1, pos2), max(pos1, pos2)

            # Get the gene to insert
            gene_to_insert = chromosome.genes[pos2]
            # Remove the gene from its original position
            chromosome.genes = np.delete(chromosome.genes, pos2)
            # Insert the gene after pos1
            chromosome.genes = np.insert(chromosome.genes, pos1 + 1, gene_to_insert)