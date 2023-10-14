from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, Value

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
        Performing tournament selection in place and in parallel fashion
        param: population - 2D array of all individuals
        param: fitness structured 2D array ["index]["fitness"]
        param: n - indicating the size of the tournament
        """

        len_population = len(population)
        # Create a shared counter
        counter = Value('i', 0)

        # Define single tournament to use parallelization
        def tournament(i):
            competitors = np.random.choice(len_population, n, replace=False)
            winner = max(competitors, key=lambda x: fitness_copy[x]["fitness"])
            selected_individual = population_copy[fitness_copy["index"][winner]]

            # Safely increment the counter and update the population
            with counter.get_lock():
                population[counter.value] = selected_individual
                fitness["index"][counter.value] = counter.value
                fitness["fitness"][counter.value] = fitness_copy["fitness"][winner]
                counter.value += 1

        # Create a copy of the population to ensure unique competitors
        population_copy = np.copy(population)
        fitness_copy = np.copy(fitness)

        # Parallelization step calling tournament
        with ThreadPoolExecutor() as executor:
            list(executor.map(tournament, range(len_population)))
