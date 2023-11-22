from numpy import ndarray

from GA import GA
from src.enums import Purpose


def two_opt_complete(ga: GA, individual: ndarray):
    """
    Applies two opt in place
    param: ga - genetic algorithm
    param: individual - structured 3D element ["individual"]["chromosome]["fitness"]
    """

    best_route = individual["chromosome"]
    # Pointer to second customer
    start_index = ga.vrp_instance.n_depots

    # Until len - 3 to guarantee 3 further edges exist
    for i in range(start_index, len(best_route) - 3):
        # Starting from i + 3 to guarantee that i and j are not adjacent edges
        for j in range(i + 3, len(best_route) - 1):

            # (i, i+1) and (j, j+1)
            edge_i = best_route[i:i + 2]
            edge_j = best_route[j:j + 2]

            new_route = best_route.copy()
            # reconnecting i -> j and i+1 -> j+1 by swapping i+1 and j
            new_route[i + 1] = edge_j[0]
            new_route[j] = edge_i[1]

            ga.decode_chromosome(new_route)
            new_fitness = ga.total_fitness

            if new_fitness < individual["fitness"]:
                individual["fitness"] = new_fitness
                individual["distance"] = ga.total_distance
                individual["time_warp"] = ga.total_time_warp
                individual["chromosome"] = new_route


def two_opt_single(ga: GA, individual: ndarray):
    """
    Applies two opt in place
    param: ga - genetic algorithm
    param: individual - structured 3D element ["individual"]["chromosome]["fitness"]
    """

    # Pointer to second customer
    start_index = ga.vrp_instance.n_depots

    # Until len - 3 to guarantee 3 further edges exist
    for i in range(start_index, len(individual["chromosome"]) - 3):
        # Starting from i + 3 to guarantee that i and j are not adjacent edges
        for j in range(i + 3, len(individual["chromosome"]) - 1):

            # (i, i+1) and (j, j+1)
            edge_i = individual["chromosome"][i:i + 2]
            edge_j = individual["chromosome"][j:j + 2]

            new_route = individual["chromosome"].copy()
            # reconnecting i -> j and i+1 -> j+1 by swapping i+1 and j
            new_route[i + 1] = edge_j[0]
            new_route[j] = edge_i[1]

            ga.decode_chromosome(new_route, Purpose.FITNESS)
            new_fitness = ga.total_fitness

            if new_fitness < individual["fitness"]:
                individual["chromosome"] = new_route
                individual["fitness"] = new_fitness
                individual["distance"] = ga.total_distance
                individual["timeout"] = ga.total_time_warp
                return


# TODO
def isp():
    pass
