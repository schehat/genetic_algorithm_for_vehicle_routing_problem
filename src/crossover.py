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
        self.LENGTH_CHROMOSOME = self.vrp_instance.n_depots + vrp_instance.n_customers

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

    def order_beginning(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Applying order crossover to second part of chromosome (Wang 2016)
        param: parent 1 and parent 2 - 1D array
        return: child 1D - array
        """

        if random.random() > self.adaptive_crossover_rate:
            return parent1

        # Ensure two distinct random positions
        pos1, pos2 = np.random.choice(range(self.vrp_instance.n_depots, self.LENGTH_CHROMOSOME), size=2, replace=False)
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
        for i in range(self.vrp_instance.n_depots, self.LENGTH_CHROMOSOME):
            if child[i] == 0:
                while parent2[insert_index] in child[self.vrp_instance.n_depots:]:
                    insert_index += 1
                child[i] = parent2[insert_index]

        return child

    def order_crossover_circular_prins(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Applying circular order crossover to the chromosome (prins 2004)
        param: parent 1 and parent 2 - 1D array
        return: child 1D - array
        """

        if random.random() > self.adaptive_crossover_rate:
            return parent1

        # Ensure two distinct random positions
        pos1, pos2 = np.random.choice(range(self.vrp_instance.n_depots, self.LENGTH_CHROMOSOME), size=2, replace=False)
        # Ensure pos1 < pos2
        pos1, pos2 = min(pos1, pos2), max(pos1, pos2)

        # Create child chromosome
        child = np.zeros_like(parent1)
        # Copy first part of chromosome
        child[:self.vrp_instance.n_depots] = parent1[:self.vrp_instance.n_depots]
        # Copy the customers within the specified range
        child[pos1:pos2 + 1] = parent1[pos1:pos2 + 1]

        # Fill the remaining positions in C1 from P2 in circular fashion. Exclude indexes for the depot information part
        insert_index = max((pos2 + 1) % self.LENGTH_CHROMOSOME, self.vrp_instance.n_depots)
        while insert_index != pos1:
            i = insert_index
            while parent2[i] in child[self.vrp_instance.n_depots:]:
                i = max((i + 1) % self.LENGTH_CHROMOSOME, self.vrp_instance.n_depots)
            child[insert_index] = parent2[i]
            insert_index = max((insert_index + 1) % self.LENGTH_CHROMOSOME, self.vrp_instance.n_depots)

        return child

    def periodic_crossover_with_insertions(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        TODO
        param: parent 1 and parent 2 - 1D array
        return: child 1D - array
        """

        if random.random() > self.adaptive_crossover_rate:
            return parent1

        # Generate a pool of unique random integers
        pool = list(range(self.vrp_instance.n_depots))
        np.random.shuffle(pool)

        # Assign integers to a1 (parent1), a2 (parent2), and a_mix for both parents
        a1 = pool[:len(pool)//3]
        a2 = pool[len(pool)//3: 2*len(pool)//3]
        a_mix = pool[2*len(pool)//3:]

        child = np.zeros_like(parent1)

        # Add visits from parent 1
        for depot_i in a1:
            n_customers = parent1[depot_i]
            start_i = np.sum(parent1[:depot_i]) + self.vrp_instance.n_depots
            end_i = start_i + n_customers

            # Copy customers from parent1 to child
            child[start_i:end_i] = parent1[start_i:end_i]

        # Add visits from parent 1 for a_mix
        for depot_i in a_mix:
            n_customers = parent1[depot_i]
            start_i = np.sum(parent1[:depot_i]) + self.vrp_instance.n_depots
            end_i = start_i + n_customers

            # Select two points i and j with uniform distribution
            i, j = np.random.choice(range(n_customers), size=2, replace=False)
            i, j = min(i, j), max(i, j)

            # Include value in start_i + i
            inserting_values = []
            inserting_values.extend(parent1[start_i: start_i+i+1])
            inserting_values.extend(parent1[start_i+j: end_i])

            # Copy selected range from parent1 to child
            child[start_i:start_i+len(inserting_values)] = inserting_values

        # Add visits from parent 2
        # TODO: check if insertion value in child is 0 then insert and break if a value != 0 is reached
        # TODO: Then split try every insertion with minimal cost and include depot n_customers adjustment and cut the 0
        for depot_i in a2:
            n_customers = parent2[depot_i]
            start_i = np.sum(parent2[:depot_i]) + self.vrp_instance.n_depots
            end_i = start_i + n_customers

            # Collect values from parent2 that are not already in the child
            inserting_values = [parent2[i] for i in range(start_i, end_i) if parent2[i] not in child]
            # Insert the values into the child at the calculated positions
            child[start_i:start_i+len(inserting_values)] = inserting_values

        # Print or use the assigned values
        print("a1:", a1)
        print("a2:", a2)
        print("a_mix:", a_mix)
