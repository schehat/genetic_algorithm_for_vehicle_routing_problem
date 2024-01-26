from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value

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


def fitness_proportional_selection(population: ndarray):
    """
    Perform fitness-proportional selection for the entire population without parallelization
    param: population - structured 3D array ["individual"]["chromosome]["fitness"]
    """

    # Calculate total fitness of the population
    total_fitness = np.sum(population["fitness"])

    # Select individuals based on roulette wheel selection
    selected_individuals = []

    for _ in range(len(population)):
        rand_value = np.random.uniform(0, total_fitness)
        cumulative_fitness = 0

        for individual in population:
            cumulative_fitness += individual["fitness"]
            if cumulative_fitness >= rand_value:
                selected_individuals.append(individual.copy())
                break

    # Convert the list to a NumPy array
    selected_individuals = np.array(selected_individuals)

    # Update the population in place
    population[:] = selected_individuals
