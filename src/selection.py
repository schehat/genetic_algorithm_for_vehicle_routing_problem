from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
from numpy import ndarray


class Selection:
    """
    Utility class providing selection methods to select from the
    current population the offsprings for the next generation
    """

    @staticmethod
    def n_tournament(population: ndarray, fitness: ndarray, n: int):
        """
        Performing tournament selection and in parallel fashion
        param: population - 2D array of all individuals
        param: fitness structured 2D array ["index]["fitness"]
        param: n - indicating the size of the tournament
        """

        len_population = len(population)

        # Define single tournament to use parallelization. Placeholder parameter necessary
        def tournament(_):
            competitors = np.random.choice(len_population, n, replace=False)
            winner = max(competitors, key=lambda x: fitness[x]["fitness"])
            return population[winner]

        # Parallelization step calling tournament(_) len_population times on different cpu cores
        with ThreadPoolExecutor() as executor:
            selected_parents = np.array(list(executor.map(tournament, range(len_population))), dtype=population.dtype)

        return selected_parents
