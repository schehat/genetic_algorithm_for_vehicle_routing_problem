import numpy as np
from numpy import ndarray


class FitnessScaling:
    """
    Utility class providing methods to scale the raw fitness value to a more suitable range for selection
    """

    def linear(self):
        pass

    @staticmethod
    def power_rank(fitness: ndarray, fitness_stats: ndarray, generation: int):
        """
        Combining rank selection and power-scaling
        param: fitness structured 2D array ["index]["fitness"]
        """

        # Sort fitness in ascending order
        fitness.sort(order='fitness')

        # Save statistics about raw fitness
        fitness_stats[generation]["max"] = np.max(fitness['fitness'])
        fitness_stats[generation]["avg"] = np.mean(fitness['fitness'])
        fitness_stats[generation]["min"] = np.min(fitness['fitness'])

        # Calculate the rank for each fitness value
        rank = np.arange(len(fitness))

        # Calculate the power of the rank with 1.005 and replace raw fitness
        fitness['fitness'] = (len(fitness) - rank) ** 1.005

        # Normalize the calculated power fitness
        sum_of_powers = np.sum(fitness['fitness'])
        fitness['fitness'] /= sum_of_powers
