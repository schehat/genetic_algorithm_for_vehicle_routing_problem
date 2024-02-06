#!/usr/bin/env python3
from datetime import datetime

import numpy as np

from GA import GA
from fitness_scaling import power_rank
from selection import n_tournaments
from local_search import two_opt
from initial_population import initial_population_grouping_savings_nnh, initial_population_random
from src.cluster import Cluster
from src.distance_measurement import broken_pairs_distance
from src.enums import Problem
from src.graph import Graph
from src.utility import update_benchmark_afvrp
from vrp import Customer, Depot, VRPInstance, ChargingStation


def read_cordeau_instance(file_path: str) -> VRPInstance:
    """
    Reads benchmark data from cordeau
    param: file_path - location to benchmark data
    return: vrp instance
    """

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the first line to get instance information
    header = lines[0].split()
    n_vehicles, n_customers, n_depots = map(int, header[1:4])
    max_duration_route, max_capacity = map(int, lines[1].split())

    customers = np.zeros((n_customers,), dtype=Customer)
    depots = np.zeros((n_depots,), dtype=Depot)
    charging_stations = np.zeros((n_depots,), dtype=ChargingStation)

    # Check which instance is checked mdvrptw or afvrp
    if not file_path.endswith("afvrp"):
        # Read customer data
        for i, line in enumerate(lines[n_depots + 1: n_depots + 1 + n_customers]):
            data = line.split()
            if len(data) >= 5:
                customer = Customer(int(data[0]), float(data[1]), float(data[2]), int(data[3]),
                                    int(data[4]), int(data[11]), int(data[12]))
                customers[i] = customer
                # print(data[0], data[1], data[2], data[3], data[4], data[11], data[12])

        # Read depot data
        for i, line in enumerate(lines[n_depots + 1 + n_customers:n_depots + 1 + n_customers + n_depots]):
            data = line.split()
            if len(data) >= 3:
                # depot id is + n_customers offset, unfavorable in later stages of GA
                depot = Depot(int(data[0]) - n_customers, float(data[1]), float(data[2]), int(data[7]), int(data[8]))
                depots[i] = depot
                # print(data[0], data[1], data[2], data[7], data[8])

        # Create Graph
        # Concatenate customer and depot coordinates to form the points array
        customer_coordinates = np.array([[customer.x, customer.y] for customer in customers])
        depot_coordinates = np.array([[depot.x, depot.y] for depot in depots])
        points = np.concatenate((customer_coordinates, depot_coordinates), axis=0)
        graph = Graph(points)

        # Update Benchmark with AFVRP data
        # k = n_depots
        # cluster = Cluster(points, k, n_depots, file_path)
        # cluster.plot_clusters()
        # n_equipments = 4
        # update_benchmark_afvrp(cluster.k_means, k, n_depots, n_equipments, file_path)

    else:
        # Read customer data
        for i, line in enumerate(lines[n_depots + 1: n_depots + 1 + n_customers]):
            data = line.split()
            if len(data) >= 5:
                customer = Customer(int(data[0]), float(data[1]), float(data[2]), int(data[3]),
                                    int(data[4]), int(data[11]), int(data[12]), label=int(data[13]), equipment=int(data[14]))
                customers[i] = customer
                # print(data[0], data[1], data[2], data[3], data[4], data[11], data[12], data[13], data[14])

        # Read depot data
        for i, line in enumerate(lines[n_depots + 1 + n_customers:n_depots + 1 + n_customers + n_depots]):
            data = line.split()
            if len(data) >= 3:
                # depot id is + n_customers offset, unfavorable in later stages of GA
                depot = Depot(int(data[0]) - n_customers, float(data[1]), float(data[2]), int(data[7]), int(data[8]),
                              label=int(data[9]))
                depots[i] = depot
                # print(data[0], data[1], data[2], data[7], data[8], data[9])

        # Read charging stations
        if file_path.endswith("afvrp"):
            for i, line in enumerate(lines[-n_depots:]):
                data = line.split()
                # depot id is + n_customers offset, unfavorable in later stages of GA
                charging_station = ChargingStation(i, float(data[1]), float(data[2]))
                charging_stations[i] = charging_station
                # print(data[0], data[1], data[2])

        # Create Graph
        # Concatenate customer and depot coordinates to form the points array
        customer_coordinates = np.array([[customer.x, customer.y] for customer in customers])
        depot_coordinates = np.array([[depot.x, depot.y] for depot in depots])
        charging_coordinates = np.array([[charging.x, charging.y] for charging in charging_stations])
        points = np.concatenate((customer_coordinates, depot_coordinates, charging_coordinates), axis=0)
        graph = Graph(points)

    if not file_path.endswith("afvrp"):
        return VRPInstance(n_vehicles, n_customers, n_depots, max_capacity, customers, depots, max_duration_route,
                           graph)
    else:
        return VRPInstance(n_vehicles, n_customers, n_depots, max_capacity, customers, depots, max_duration_route,
                           graph, charging_stations)


if __name__ == "__main__":
    # Set the print options to control the display format
    np.set_printoptions(threshold=np.inf)

    """
    GA CONFIGURATION
    """
    # Create vrp instance
    INSTANCE_NAME = "pr01_afvrp"
    INSTANCE_FILE_PATH = f"../benchmark/c-mdvrptw/{INSTANCE_NAME}"
    VRP_INSTANCE = read_cordeau_instance(INSTANCE_FILE_PATH)
    # Problem type needs to match to instance name. For AFVRP testing instance name suffix is with _afvrp
    PROBLEM_TYPE = Problem.AFVRP
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_prefix_name = f"../BA_results/{INSTANCE_NAME}_hybrid/{time_stamp}"
    # All hybrid elements active. For separate testing changes in GA necessary
    HYBRID = True

    # Set GA parameters
    POPULATION_SIZE = 100
    MAX_GENERATIONS = 1000
    INITIAL_POPULATION = initial_population_grouping_savings_nnh
    # INITIAL_POPULATION = initial_population_random
    FITNESS_SCALING = power_rank
    SELECTION_METHOD = n_tournaments
    LOCAL_SEARCH_METHOD = two_opt
    DISTANCE_METHOD = broken_pairs_distance

    ga = GA(VRP_INSTANCE,
            POPULATION_SIZE,
            MAX_GENERATIONS,
            INITIAL_POPULATION,
            FITNESS_SCALING,
            SELECTION_METHOD,
            LOCAL_SEARCH_METHOD,
            DISTANCE_METHOD,
            PROBLEM_TYPE,
            file_prefix_name,
            HYBRID)

    ga.run()
