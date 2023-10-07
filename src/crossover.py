import random

import numpy as np

from chromosome import Chromosome


class Crossover:
    adaptive_crossover_rate = 0.5 #TODO

    def __init__(self, CROSSOVER_RATE: float):
        self.CROSSOVER_RATE = CROSSOVER_RATE

    # Recombination first part of chromosome
    def uniform(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        # Check if crossover should occur
        if random.random() > self.CROSSOVER_RATE:
            return parent1  # No crossover, return parent1 as is

        num_genes = Chromosome.N_DEPOTS * Chromosome.N_VEHICLES_PER_DEPOT

        child_genes = parent1.genes.copy()

        # Iterate through the genes and apply uniform crossover
        for i in range(num_genes):
            if random.random() <= self.adaptive_crossover_rate:
                child_genes[i] = parent2.genes[i]

        child = Chromosome(child_genes)
        return child
    
    def order(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        # Check if crossover should occur
        if random.random() > self.CROSSOVER_RATE:
            return parent1  # No crossover, return parent1 as is

        start = Chromosome.N_DEPOTS * Chromosome.N_VEHICLES_PER_DEPOT
        end = start + Chromosome.N_CUSTOMERS

        # Ensure two distinct random positions
        pos1, pos2 = np.random.choice(range(start, end), size=2, replace=False)
        # Ensure pos1 < pos2
        pos1, pos2 = min(pos1, pos2), max(pos1, pos2)
        print(f"{pos1} {pos2}")

        # Create child chromosome 
        child_genes = np.zeros_like(parent1.genes)
        # Copy first part of chromosome
        child_genes[:start] = parent1.genes[:start]
        # Copy the values within the specified range
        child_genes[pos1:pos2 + 1] = parent1.genes[pos1:pos2 + 1]

        # Fill in the missing genes from parent2
        insert_index = start
        for i in range(start, end):
            if child_genes[i] == 0:
                while parent2.genes[insert_index] in child_genes[start:end]:
                    insert_index += 1
                child_genes[i] = parent2.genes[insert_index]

        child = Chromosome(child_genes)
        return child






