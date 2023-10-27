import numpy as np
from numpy import ndarray

from vrp import VRPInstance


def linear(self):
    pass


def two_opt(ga: any, individual: ndarray):
    """
    TODO
    """

    best_route = individual["chromosome"]
    # Pointer to second customer
    start_index = ga.vrp_instance.n_depots + ga.vrp_instance.n_vehicles

    # Until len - 3 to guarantee 3 further edges exist
    for i in range(start_index, len(best_route) - 3):
        # Starting from i + 3 to guarantee that i and j are not adjacent edges
        for j in range(i + 3, len(best_route) - 1):

            # (i, i+1) and (j, j+1)
            edge_i = best_route[i:i+2]
            edge_j = best_route[j:j+2]

            new_route = best_route.copy()
            # reconnecting i -> j and i+1 -> j+1 by swapping i+1 and j
            new_route[i+1] = edge_j[0]
            new_route[j] = edge_i[1]

            new_fitness = ga.evaluate_fitness(new_route)

            if new_fitness < individual["fitness"]:
                best_route = new_route
                individual["fitness"] = new_fitness
