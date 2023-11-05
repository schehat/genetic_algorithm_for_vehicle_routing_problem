import random

import numpy as np

from vrp import VRPInstance


class Crossover:
    """
    Genetic operator to recombine genetic information of parents to produce a child
    """

    # Will be set from the GA
    adaptive_crossover_rate: float = None

    def __init__(self, vrp_instance: VRPInstance, crossover_rate: float):
        self.vrp_instance: VRPInstance = vrp_instance
        self.CROSSOVER_RATE = crossover_rate
        # helper attributes
        self.END_SECOND_PART = self.vrp_instance.n_depots + vrp_instance.n_customers

    def uniform(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Applying uniform crossover to first part of chromosome
        param: parent 1 and parent 2 - 1D array
        return: child - 1D array
        """

        # Check if crossover should occur
        if random.random() > self.adaptive_crossover_rate:
            return parent1  # No crossover, return parent1 as is

        child = parent1.copy()

        # Iterate through the depots and apply uniform crossover
        for i in range(self.vrp_instance.n_depots):
            if random.random() <= self.CROSSOVER_RATE:
                child[i] = parent2[i]

        return child

    def order(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Applying order crossover to second part of chromosome
        param: parent 1 and parent 2 - 1D array
        return: child 1D - array
        """

        if random.random() > self.adaptive_crossover_rate:
            return parent1

        # Ensure two distinct random positions
        pos1, pos2 = np.random.choice(range(self.vrp_instance.n_depots, self.END_SECOND_PART), size=2, replace=False)
        # Ensure pos1 < pos2
        pos1, pos2 = min(pos1, pos2), max(pos1, pos2)

        # Create child chromosome 
        child = np.zeros_like(parent1)
        # Copy first part of chromosome
        child[:self.vrp_instance.n_depots] = parent1[:self.vrp_instance.n_depots]
        # Copy the customers within the specified range
        child[pos1:pos2 + 1] = parent1[pos1:pos2 + 1]

        # Fill in the missing customers from parent2
        insert_index = self.vrp_instance.n_depots
        for i in range(self.vrp_instance.n_depots, self.END_SECOND_PART):
            if child[i] == 0:
                while parent2[insert_index] in child[self.vrp_instance.n_depots:self.END_SECOND_PART]:
                    insert_index += 1
                child[i] = parent2[insert_index]

        return child
