import numpy as np
from numpy.core.records import ndarray


def broken_pairs_distance(chromosome_a: ndarray, chromosome_b: ndarray, n_depots) -> int:
    """
    Counts number of paris in chromosome_a separated in chromosome_b as diversity distance measurement (Prins 2009)
    param: chromosome_a, chromosome b
    return: number of broken pairs
    """

    # Create a mapping of customers to their indices in chromosome_b
    index_mapping = {customer: i for i, customer in enumerate(chromosome_b[n_depots:])}
    broken_pairs_count = 0
    n = len(chromosome_a[n_depots:])

    for i in range(n - 1):
        customer_a1, customer_a2 = chromosome_a[n_depots + i], chromosome_a[n_depots + i + 1]

        # Find the positions of the customers in B
        index_b1, index_b2 = index_mapping[customer_a1], index_mapping[customer_a2]

        # Check if the pair is broken in chromosome_b
        if (index_b1 + 1) % n != index_b2:
            broken_pairs_count += 1

    # NOT CLEAN HAMMING FOR DEPOT INFORMATION
    for i in range(n_depots):
        customer_a_i = chromosome_a[i]
        customer_b_i = chromosome_b[i]

        # Check if the elements at the same position are different
        if customer_a_i != customer_b_i:
            broken_pairs_count += abs(customer_a_i - customer_b_i)

    return broken_pairs_count


def hamming_distance(chromosome_a: ndarray, chromosome_b: ndarray) -> int:
    """
    Calculates hamming distance between 2 chromosomes as the number of not alike genes
    param: chromosome_a, chromosome b
    return: number of not alike genes
    """

    hamming_distance = 0
    for i in range(len(chromosome_a)):
        customer_a_i = chromosome_a[i]
        customer_b_i = chromosome_b[i]

        # Check if the elements at the same position are different
        if customer_a_i != customer_b_i:
            hamming_distance += 1

    return hamming_distance


class EuclideanDistance:
    def __init__(self, ga: "GA"):
        self.shortest_paths_cache = {}

    def euclidean_distance(self, obj1, obj2) -> float:
        """
        Calculate fitness for a single chromosome
        param: obj1 and obj2 - Customers or Depots
        return: distance
        """
        obj1_cord = (obj1.x, obj1.y)
        obj2_cord = (obj2.x, obj2.y)
        if (obj1_cord, obj2_cord) in self.shortest_paths_cache:
            # If the result is already cached, return it
            return self.shortest_paths_cache[(obj1_cord, obj2_cord)]

        dx = obj1.x - obj2.x
        dy = obj1.y - obj2.y
        distance = (dx**2 + dy**2)**0.5
        self.shortest_paths_cache[(obj1_cord, obj2_cord)] = distance

        return distance
