from random import random, shuffle
from typing import Tuple

import numpy as np

from vrp import VRPInstance


class Crossover:
    """
    Genetic operator to recombine genetic information of parents to produce a child
    """

    UNIFORM_RATE = 0.5

    def __init__(self, vrp_instance: VRPInstance):
        self.vrp_instance: VRPInstance = vrp_instance
        self.LENGTH_CHROMOSOME = self.vrp_instance.n_depots + vrp_instance.n_customers

    def uniform(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Applying uniform crossover to first part of chromosome
        param: parent 1 and parent 2 - 1D array
        return: child - 1D array
        """

        child = parent1.copy()

        # Iterate through the depots and apply uniform crossover
        for i in range(self.vrp_instance.n_depots):
            if random() <= self.UNIFORM_RATE:
                child[i] = parent2[i]

        return child

    def order_beginning(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Applying order crossover to second part of chromosome (Wang 2016)
        param: parent 1 and parent 2 - 1D array
        return: child 1D - array
        """

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
        Applying circular order crossover to the chromosome (Prins 2004)
        param: parent 1 and parent 2 - 1D array
        return: child 1D - array
        """

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

    def periodic_crossover_with_insertions(self, parent1: np.ndarray, parent2: np.ndarray, ga: "GA", fitness:float) -> Tuple[np.ndarray, float]:
        """
        Advanced crossover inspired by (Vidal 2012)
        param: parent 1 and parent 2 - 1D array
        return: child 1D - array
        """

        # Generate a pool of unique random integers
        pool = list(range(self.vrp_instance.n_depots))
        np.random.shuffle(pool)

        # Assign integers to a1 (parent1), a2 (parent2), and a_mix for both parents
        a1 = pool[:len(pool) // 3]
        a2 = pool[len(pool) // 3: 2 * len(pool) // 3]
        a_mix = pool[2 * len(pool) // 3:]

        child = np.zeros_like(parent1)

        # Add visits from parent 1
        for depot_i in a1:
            n_customers = parent1[depot_i]
            start_i = np.sum(parent1[:depot_i]) + ga.vrp_instance.n_depots
            end_i = start_i + n_customers

            # Copy customers from parent1 to child
            child[start_i:end_i] = parent1[start_i:end_i]

        # Add visits from parent 1 for a_mix
        for depot_i in a_mix:
            n_customers = parent1[depot_i]
            start_i = np.sum(parent1[:depot_i]) + ga.vrp_instance.n_depots
            end_i = start_i + n_customers

            # Select two points i and j with uniform distribution
            if n_customers > 1:
                i, j = np.random.choice(range(n_customers), size=2, replace=False)
                i, j = min(i, j), max(i, j)
            else:
                i = 0

            # Include value in start_i + i
            inserting_values = []
            inserting_values.extend(parent1[start_i: start_i + i + 1])
            # Skip right part of point (customers after j) if only 1 customer exists
            if i != 0:
                inserting_values.extend(parent1[start_i + j: end_i])

            # Copy selected range from parent1 to child
            child[start_i:start_i + len(inserting_values)] = inserting_values

        # Add visits from parent 2
        a_mix.extend(a2)
        np.random.shuffle(a_mix)
        for depot_i in a_mix:
            n_customers = parent2[depot_i]
            start_i = np.sum(parent2[:depot_i]) + ga.vrp_instance.n_depots
            end_i = start_i + n_customers

            # Collect values from parent2 that are not already in the child
            inserting_values = [parent2[i] for i in range(start_i, end_i) if parent2[i] not in child]

            # Adjust indexes to parent1 to match correct depot range
            n_customers = parent1[depot_i]
            start_i = np.sum(parent1[:depot_i]) + ga.vrp_instance.n_depots
            end_i = start_i + n_customers

            for i in range(end_i - start_i):
                # Exhausted all values or stop if confronting depot information
                if len(inserting_values) == 0 or end_i - i - 1 < ga.vrp_instance.n_depots:
                    break
                # Insert at the end and decrease every step
                if child[end_i - i - 1] == 0:
                    # To not change order of inserting_values insert in reversed order
                    child[end_i - i - 1] = inserting_values.pop()
                else:
                    # Stop inserting even if not all values are exhausted
                    break

        for depot_i in range(ga.vrp_instance.n_depots):
            n_customers = parent1[depot_i]

            start_i = np.sum(parent1[:depot_i]) + ga.vrp_instance.n_depots
            end_i = start_i + n_customers

            # Count non-zero values in the specified range
            customer_count = np.count_nonzero(child[start_i:end_i] != 0)

            # Assign the count to the corresponding depot index in the child
            child[depot_i] = customer_count

        # Extract the segment related to customers
        customer_segment = child[ga.vrp_instance.n_depots:]
        # Find the non-zero indices in the customer segment
        non_zero_indices = customer_segment != 0
        # Update the customer segment with only non-zero values
        customer_segment = customer_segment[non_zero_indices]
        # Update the child array with the modified customer segment
        child = np.concatenate((child[:ga.vrp_instance.n_depots], customer_segment))

        all_customers = list(range(1, ga.vrp_instance.n_customers + 1))
        existing_customers = list(child[ga.vrp_instance.n_depots:])
        missing_customers = np.setdiff1d(all_customers, existing_customers)
        best_fitness = fitness
        max_improvements = 2
        max_depot_iterations = 3
        for m_customer in missing_customers:
            best_position = ga.vrp_instance.n_depots
            depot_assignment = 0
            n_improvements = 0

            customer_offset = ga.vrp_instance.n_depots

            for depot_i in range(ga.vrp_instance.n_depots):
                depot_n_customers = child[depot_i]

                # Range +1 to also insert after last index
                shuffle_insertion = list(range(depot_n_customers + 1))
                shuffle(shuffle_insertion)
                for customer_i in shuffle_insertion[:max_depot_iterations]:

                    # Enable only insert position after depot range if last depot
                    if customer_i == depot_n_customers and depot_i != ga.vrp_instance.n_depots - 1:
                        continue

                    temp_child = child.copy()
                    temp_child = np.insert(temp_child, customer_offset + customer_i, m_customer)
                    temp_child[depot_i] += 1
                    p_complete = ga.split.split(temp_child)[0]

                    zero_indices = np.where(p_complete == 0)[0]
                    selected_values = p_complete[np.concatenate([zero_indices - 1])]
                    total_fitness = np.sum(selected_values)

                    if total_fitness < best_fitness:
                        best_fitness = total_fitness
                        best_position = customer_offset + customer_i
                        depot_assignment = depot_i
                        n_improvements += 1
                        if n_improvements >= max_improvements:
                            break

                customer_offset += depot_n_customers

            # Add m_customer at best_position final state
            child = np.insert(child, best_position, m_customer)
            # Increment the customer count for the corresponding depot
            child[depot_assignment] += 1

        return child, best_fitness
