#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

from GA import GA
from fitness_scaling import power_rank
from selection import n_tournaments
from local_search import two_opt
from initial_population import initial_population_grouping_savings_nnh, initial_population_random
from src.distance_measurement import broken_pairs_distance
from vrp import Customer, Depot, VRPInstance
from sklearn.cluster import KMeans


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

    # Read customer data
    for i, line in enumerate(lines[n_depots + 1: n_depots + 1 + n_customers]):
        data = line.split()
        if len(data) >= 5:
            customer = Customer(int(data[0]), float(data[1]), float(data[2]), int(data[3]),
                                int(data[4]), int(data[-2]), int(data[-1]))
            customers[i] = customer

    # Read depot data
    for i, line in enumerate(lines[n_depots + 1 + n_customers:]):
        data = line.split()
        if len(data) >= 3:
            # depot id is + n_customers offset, unfavorable in later stages of GA
            depot = Depot(int(data[0]) - n_customers, float(data[1]), float(data[2]), int(data[-2]), int(data[-1]))
            depots[i] = depot

    customer_coordinates = np.array([[customer.x, customer.y] for customer in customers])
    depot_coordinates = np.array([[depot.x, depot.y] for depot in depots])
    points = np.concatenate((customer_coordinates, depot_coordinates), axis=0)

    # Specify the number of clusters (k)
    k = n_depots

    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=k)

    # Fit the model to the data
    kmeans.fit(points)

    # Get the cluster centers and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Separate customer and depot points
    customer_points = points[:-n_depots]
    depot_points = points[-n_depots:]

    # Plot the customer positions with circle markers and colors based on cluster labels
    plt.scatter(customer_points[:, 0], customer_points[:, 1], c=labels[:-n_depots], cmap='viridis', label='Customers', marker='o')

    # Plot the depot positions with bigger circles, black edge, and colors based on cluster labels
    plt.scatter(depot_points[:, 0], depot_points[:, 1], c=labels[-n_depots:], cmap='viridis', label='Depots', marker='o', s=100, edgecolors='black')

    # Plot the cluster centers with red 'X' markers
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', label='Cluster Centers')

    print("Cluster Centers:")
    print(centroids)
    print("\nLabels:")
    print(labels)

    plt.title('Customer and Depot Clustering with K-Means')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

    return VRPInstance(n_vehicles, n_customers, n_depots, max_capacity, customers, depots, max_duration_route)


if __name__ == "__main__":
    # Set the print options to control the display format
    np.set_printoptions(threshold=np.inf)

    # Create vrp instance
    INSTANCE_NAME = "pr01"
    INSTANCE_FILE_PATH = f"../benchmark/c-mdvrptw/{INSTANCE_NAME}"
    VRP_INSTANCE = read_cordeau_instance(INSTANCE_FILE_PATH)

    # Set GA parameters
    POPULATION_SIZE = 50
    CROSSOVER_RATE = 0.5
    MUTATION_RATE = 0.5
    MAX_GENERATIONS = 500
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
            INSTANCE_NAME)

    ga.run()
