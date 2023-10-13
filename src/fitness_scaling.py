import numpy as np
from numpy import ndarray


class FitnessScaling:
    """
    Utility class providing methods to scale the raw fitness value to a more suitable range for selection
    """

    def linear(self):
        pass

    @staticmethod
    def power_rank(fitness: ndarray):
        """
        Combining rank selection and power-scaling
        param: fitness structured 2D array ["index]["fitness"]
        """

        fitness.sort(order='fitness')

        # Calculate the rank for each fitness value
        rank = np.arange(len(fitness))

        # Calculate the power of the rank with 1.005 and replace raw fitness
        fitness['fitness'] = (rank + 1) ** 1.005

        # Normalize the calculated power fitness
        sum_of_powers = np.sum(fitness['fitness'])
        fitness['fitness'] /= sum_of_powers
