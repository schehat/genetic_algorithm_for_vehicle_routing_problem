import os

import numpy as np

import matplotlib.pyplot as plt
from numpy import ndarray

from vrp import Depot, Customer


def plot_fitness(ga, width=8, height=6, interval=50):
    """
    Plot data points for minimum, average, and maximum fitness values over generations at given intervals
    param: ga - genetic algorithm
    param: width and height - size of figure
    interval: entries between data point plots
    """
    plt.figure(figsize=(width, height))

    x = np.arange(ga.max_generations)
    min_fitness = ga.fitness_stats["min"]
    avg_fitness = ga.fitness_stats["avg"]

    # Plot data points at 50-generation intervals
    plt.plot(x[::interval], min_fitness[::interval], marker='o', label='Min Fitness')
    plt.plot(x[::interval], avg_fitness[::interval], marker='o', label='Avg Fitness')

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.grid(True)
    plt.legend()

    save_plot(plt, f"../results/{ga.__class__.__name__}/{ga.TIMESTAMP}", f"fitness")

    plt.show()


def plot_routes(ga, chromosome: ndarray, width=8, height=6):
    """
    Plot the routes for a given solution
    param: ga - genetic algorithm
    param: chromosome - the best solution
    param: width and height - size of figure
    """

    # fitness: total distance
    depot_index = 0
    vehicle_index = ga.vrp_instance.n_depots
    customer_index = ga.vrp_instance.n_depots + ga.vrp_instance.n_vehicles
    # keep track of iterations of a depot
    depot_value_counter = 1

    # TODO plot depots for every rout
    colors = ["red", "green", "blue", "orange"]

    for i in range(ga.vrp_instance.n_vehicles):
        vehicle_i_n_customers = chromosome[vehicle_index + i]
        # Capacity for every vehicle the same at the moment. TODO dynamic capacity which vehicle class
        vehicle_i_capacity = 0

        # Check if all iterations for vehicles of current depot are done. Then continue with next depot
        if depot_value_counter > chromosome[depot_index]:
            depot_value_counter = 1
            depot_index += 1

        vehicle_i_depot: Depot = ga.vrp_instance.depots[depot_index]

        # Storing the routes for plotting
        x = []
        y = []

        for j in range(vehicle_i_n_customers):
            customer_value1 = chromosome[customer_index + j]
            # Indexing of customers starts with 1 not 0, so -1 necessary
            customer_1: Customer = ga.vrp_instance.customers[customer_value1 - 1]

            # First iteration in loop: first trip
            if j == 0:
                # Assuming single customer demand <= vehicle max capacity

                # Add routing points
                x.append(vehicle_i_depot.x)
                x.append(customer_1.x)
                y.append(vehicle_i_depot.y)
                y.append(customer_1.y)
                # TODO add capacity constraint meaning vehicles with different capacity
                # Thus customer demand > vehicle max capacity possible but at least 1 vehicle exists with greater capacity
                vehicle_i_capacity += customer_1.demand

            # Check if next customer exists in route exists
            if j < vehicle_i_n_customers - 1:
                customer_value2 = chromosome[customer_index + j + 1]
                customer_2: Customer = ga.vrp_instance.customers[customer_value2 - 1]

                # Check customer_2 demand exceeds vehicle capacity limit
                # TODO Add heterogeneous capacity for vehicles
                if vehicle_i_capacity + customer_2.demand > ga.vrp_instance.max_capacity:
                    # Trip back to depot necessary. Assuming heading back to same depot it came from
                    # TODO visit different depot if possible e.g. AF-VRP charging points for robots

                    x.append(vehicle_i_depot.x)
                    x.append(customer_2.x)
                    y.append(vehicle_i_depot.y)
                    y.append(customer_2.y)

                    # from depot to next customer
                    vehicle_i_capacity = 0
                else:
                    # Add distance between customers
                    x.append(customer_2.x)
                    y.append(customer_2.y)

                vehicle_i_capacity += customer_2.demand

            # Last iteration in loop, add trip from last customer to depot
            if j >= vehicle_i_n_customers - 1:
                x.append(vehicle_i_depot.x)
                y.append(vehicle_i_depot.y)

        customer_index += vehicle_i_n_customers
        depot_value_counter += 1

        plt.figure(figsize=(width, height))
        for index in range(ga.vrp_instance.n_depots):
            depot: Depot = ga.vrp_instance.depots[index]
            plt.scatter(depot.x, depot.y, s=100, color=colors[index], label=f'Depot {index + 1}', zorder=2)

        plt.plot(x, y, marker='o', color=colors[i], zorder=1)

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Routes Visualization Vehicle {i + 1}')
        plt.grid(True)
        plt.legend()

        save_plot(plt, f"../results/{ga.__class__.__name__}/{ga.TIMESTAMP}", f"route_vehicle_{i + 1}")

        plt.show()


def save_plot(plotting, location: str, file_name: str):
    os.makedirs(location, exist_ok=True)
    file_name = os.path.join(location, file_name)
    plotting.savefig(file_name)
