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

    def periodic_crossover_with_insertions(self, parent1: np.ndarray, parent2: np.ndarray, ga: "GA") -> np.ndarray:
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
        a_mix.extend(a2)
        np.random.shuffle(a_mix)
        for depot_i in a_mix:
            n_customers = parent2[depot_i]
            start_i = np.sum(parent2[:depot_i]) + self.vrp_instance.n_depots
            end_i = start_i + n_customers

            # Collect values from parent2 that are not already in the child
            inserting_values = [parent2[i] for i in range(start_i, end_i) if parent2[i] not in child]

            # Adjust indexes to parent1 to match correct depot range
            n_customers = parent1[depot_i]
            start_i = np.sum(parent1[:depot_i]) + self.vrp_instance.n_depots
            end_i = start_i + n_customers

            for i in range(end_i - start_i):
                # Exhausted all values
                if len(inserting_values) == 0:
                    break
                # Insert at the end and decrease every step
                if child[end_i - i - 1] == 0:
                    # To not change order of inserting_values insert in reversed order
                    child[end_i - i - 1] = inserting_values.pop()
                else:
                    # Stop inserting even if not all values are exhausted
                    break

        for depot_i in range(self.vrp_instance.n_depots):
            n_customers = parent1[depot_i]

            start_i = np.sum(parent1[:depot_i]) + self.vrp_instance.n_depots
            end_i = start_i + n_customers

            # Count non-zero values in the specified range
            customer_count = np.count_nonzero(child[start_i:end_i] != 0)

            # Assign the count to the corresponding depot index in the child
            child[depot_i] = customer_count

        non_zero_indices = child != 0
        child = child[non_zero_indices]

        all_customers = list(range(1, self.vrp_instance.n_customers + 1))
        existing_customers = list(child[self.vrp_instance.n_depots:])
        missing_customers = np.setdiff1d(all_customers, existing_customers)

        for m_customer in missing_customers:
            best_fitness = float("inf")
            best_position = None

            for i in range(self.vrp_instance.n_depots, len(child) + 1):
                temp_child = child.copy()
                temp_child = np.insert(temp_child, i, m_customer)

                ga.split.split(temp_child)

                zero_indices = np.where(ga.p_complete == 0)[0]
                selected_values = ga.p_complete[np.concatenate([zero_indices - 1])]
                ga.total_fitness = np.sum(selected_values)

                if ga.total_fitness < best_fitness:
                    best_fitness = ga.total_fitness
                    best_position = i

            # Add m_customer at best_position final state
            child = np.insert(child, best_position, m_customer)



