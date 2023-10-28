from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, Value

import numpy as np
from numpy import ndarray


def n_tournaments(population: ndarray, n: int):
    """
    Performing tournament selection in place and in parallel fashion
    param: population - structured 3D array ["individual"]["chromosome]["fitness"]
    param: n - size of tournaments
    """

    # Create a copy of the population to ensure fair tournament to pick competitors from
    population_copy = np.copy(population)
    len_population = len(population)

    # Create a shared counter
    counter = Value('i', 0)

    # Define single tournament to use parallelization
    def tournament(i):
        # Select the winner based on fitness
        competitors = np.random.choice(len_population, n, replace=False)
        winner = min(competitors, key=lambda x: population_copy[x]["fitness"])
        selected_individual = population_copy[winner]

        # Safely increment the counter and update the population
        with counter.get_lock():
            population[counter.value] = selected_individual
            counter.value += 1

    # Parallelization step calling tournament
    with ThreadPoolExecutor() as executor:
        list(executor.map(tournament, range(len_population)))
