import numpy as np
from numpy import ndarray


class FitnessScaling:
    def linear(self):
        pass

    @staticmethod
    def power_rank(fitness: ndarray):
        fitness.sort(order='fitness')

        # Calculate the rank (actual index) for each fitness value
        rank = np.arange(len(fitness))

        # Calculate the power of the rank with 1.005 and replace raw fitness
        fitness['fitness'] = (rank + 1) ** 1.005

        # Calculate sum of all powers in parallel
        sum_of_powers = np.sum(fitness['fitness'])

        # Normalize the calculated power fitness
        fitness['fitness'] /= sum_of_powers
