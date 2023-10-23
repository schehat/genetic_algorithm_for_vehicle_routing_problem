import numpy as np
from numpy import ndarray


def linear():
    pass


def power_rank(fitness_scores: ndarray, exponent: float = 1.005):
    """
    Combining rank selection and power-scaling for more robust behavior and fast convergence.
    param: fitness structured 2D array ["index]["fitness"]
    param: exponent - control parameter for selection pressure. In literature 1.005 is recommended
    """

    # Sort fitness in ascending order
    fitness_scores.sort(order='fitness')

    # Calculate the rank for each fitness value
    rank = np.arange(len(fitness_scores))

    # Calculate the power of the rank and replace raw fitness
    fitness_scores['fitness'] = (len(fitness_scores) - rank) ** exponent

    # Normalize the calculated power fitness
    sum_of_powers = np.sum(fitness_scores['fitness'])
    fitness_scores['fitness'] /= sum_of_powers
