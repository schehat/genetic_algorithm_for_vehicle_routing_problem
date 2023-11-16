from numpy.core.records import ndarray


def broken_pairs_distance(chromosome_a: ndarray, chromosome_b: ndarray):
    # Create a mapping of customers to their indices in chromosome_b
    index_mapping = {customer: i for i, customer in enumerate(chromosome_b)}
    broken_pairs_count = 0
    n = len(chromosome_a)

    for i in range(n - 1):
        customer_a1, customer_a2 = chromosome_a[i], chromosome_a[i + 1]

        # Find the positions of the customers in B
        index_b1, index_b2 = index_mapping[customer_a1], index_mapping[customer_a2]

        # Check if the pair is broken in chromosome_b
        if (index_b1 + 1) % n != index_b2:
            broken_pairs_count += 1

    return broken_pairs_count


def hamming_distance(chromosome_a: ndarray, chromosome_b: ndarray) -> float:
    hamming_distance = 0
    for i in range(len(chromosome_a)):
        customer_a_i = chromosome_a[i]
        customer_b_i = chromosome_b[i]

        # Check if the elements at the same position are different
        if customer_a_i != customer_b_i:
            hamming_distance += 1

    return hamming_distance
