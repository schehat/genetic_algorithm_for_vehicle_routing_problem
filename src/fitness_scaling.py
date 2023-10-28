import numpy as np
from numpy import ndarray


def linear():
    pass


def power_rank(population: ndarray, exponent: float = 1.005):
    """
    Combining rank selection and power-scaling for more robust behavior and fast convergence
    param: population - structured 3D array ["individual"]["chromosome"]["fitness"]
    param: exponent - control parameter for selection pressure. In literature 1.005 is recommended
    """

    # Sort the population based on fitness values
    sorted_indices = np.argsort(population["fitness"])
    population[:] = population[sorted_indices]

    # Calculate the rank for each fitness value
    rank = np.arange(len(population))

    # Calculate the power of the rank
    population["fitness"] = (rank + 1) ** exponent

    # Normalize the calculated power fitness
    sum_of_powers = np.sum(population["fitness"])
    population["fitness"] /= sum_of_powers
