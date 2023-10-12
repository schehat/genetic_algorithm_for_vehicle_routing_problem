import random

import numpy as np

from vrp_instance import VRPInstance


class Crossover:
    adaptive_crossover_rate = 0.5 #TODO

    def __init__(self, vrp_instance: VRPInstance, CROSSOVER_RATE: float):
        self.vrp_instance: VRPInstance = vrp_instance
        self.START_SECOND_PART = vrp_instance.n_depots
        self.START_THIRD_PART = self.START_SECOND_PART + vrp_instance.n_vehicles
        self.END_THIRD_PART = self.START_THIRD_PART + vrp_instance.n_customers
        self.CROSSOVER_RATE = CROSSOVER_RATE

    # first and second part of chromosome
    def uniform(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        # Check if crossover should occur
        if random.random() > self.CROSSOVER_RATE:
            return parent1  # No crossover, return parent1 as is

        child = parent1.copy()

        # Iterate through the depots and apply uniform crossover
        for i in range(self.START_THIRD_PART):
            if random.random() <= self.adaptive_crossover_rate:
                child[i] = parent2[i]

        # Iterate through the vehicles and apply uniform crossover
        for i in range(self.START_THIRD_PART, self.START_THIRD_PART):
            if random.random() <= self.adaptive_crossover_rate:
                child[i] = parent2[i]

        return child
    
    # third part of chromosome
    def order(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        # Check if crossover should occur
        if random.random() > self.CROSSOVER_RATE:
            return parent1  # No crossover, return parent1 as is

        # Ensure two distinct random positions
        pos1, pos2 = np.random.choice(range(self.START_THIRD_PART, self.END_THIRD_PART), size=2, replace=False)
        # Ensure pos1 < pos2
        pos1, pos2 = min(pos1, pos2), max(pos1, pos2)
        print(f"{pos1} {pos2}")

        # Create child chromosome 
        child = np.zeros_like(parent1)
        # Copy first part of chromosome
        child[:self.START_THIRD_PART] = parent1[:self.START_THIRD_PART]
        # Copy the customers within the specified range
        child[pos1:pos2 + 1] = parent1[pos1:pos2 + 1]

        # Fill in the missing customers from parent2
        insert_index = self.START_THIRD_PART
        for i in range(self.START_THIRD_PART, self.END_THIRD_PART):
            if child[i] == 0:
                while parent2[insert_index] in child[self.START_THIRD_PART:self.END_THIRD_PART]:
                    insert_index += 1
                child[i] = parent2[insert_index]

        return child