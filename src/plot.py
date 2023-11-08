import os

import matplotlib
import numpy as np

import matplotlib.pyplot as plt
from numpy import ndarray

from enums import Purpose
from GA import GA
from vrp import Depot, Customer


def plot_fitness(ga: GA, width=8, height=6, interval=50):
    """
    Plot data points for minimum and average fitness values over generations at given intervals
    param: ga - genetic algorithm
    param: width and height - size of figure
    param: interval - entries between data point plots
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


def plot_routes(ga, individual: ndarray, width=12, height=10):
    """
    Plot the routes for a given chromosome solution
    param: ga - genetic algorithm
    param: chromosome - structured 3D element ["individual"]["chromosome]["fitness"]
    param: width and height - size of figure
    """

    colors = ["salmon", "gold", "lightgreen", "mediumslateblue", "indianred", "orange", "limegreen", "deepskyblue",
              "yellow", "turquoise", "dodgerblue", "violet", "peru", "springgreen", "steelblue", "crimson"]

    # While decoding chromosome use collect_routes
    ga.decode_chromosome(individual, Purpose.PLOTTING)

    # Plot for every depot it routes
    color_index = 0
    for depot_index in range(ga.vrp_instance.n_depots):
        plt.figure(figsize=(width, height))

        # plot depots
        for index in range(ga.vrp_instance.n_depots):
            depot: Depot = ga.vrp_instance.depots[index]
            plt.scatter(depot.x, depot.y, s=150, color=colors[index], edgecolor='black', label=f'Depot {index + 1}',
                        zorder=2)

        # Plot routes
        depot = ga.vrp_instance.depots[depot_index]
        x_pos = ga.route_data[depot_index]["x_pos"]
        y_pos = ga.route_data[depot_index]["y_pos"]
        customer_ids = ga.route_data[depot_index]["customer_ids"]

        # Initialize variables to track the current route
        start = 0
        # start loop from 1 to exclude depot
        for j in range(1, len(x_pos)):
            # Start a new route with a new color
            if x_pos[j] == depot.x and y_pos[j] == depot.y:
                # Plot the current route
                plt.plot(x_pos[start:j+1], y_pos[start:j+1], marker='o', color=colors[color_index], zorder=1)
                if color_index < len(colors):
                    color_index += 1
                else:
                    color_index = 0
                # Should point to depot
                start = j

        # Add customer ids and order of iteration as labels to the plot
        for j in range(len(x_pos)):
            plt.text(x_pos[j], y_pos[j], customer_ids[j], fontsize=8, ha='right')
            if j < len(x_pos) - 1:
                mid_x = (x_pos[j] + x_pos[j + 1]) / 2
                mid_y = (y_pos[j] + y_pos[j + 1]) / 2
                plt.text(mid_x, mid_y, f'{j + 1}', fontsize=8, ha='center', va='center',
                         bbox=dict(boxstyle='round', pad=0.1, edgecolor='black', facecolor='#ffffff'))

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Depot Route Visualization {depot_index + 1}')
        plt.grid(True)
        plt.legend()

        save_plot(plt, f"../results/{ga.__class__.__name__}/{ga.TIMESTAMP}", f"depot_route{depot_index + 1}")

        plt.show()

    plt.figure(figsize=(width, height))
    # plot depots
    for index in range(ga.vrp_instance.n_depots):
        depot: Depot = ga.vrp_instance.depots[index]
        plt.scatter(depot.x, depot.y, s=150, color=colors[index], edgecolor='black', label=f'Depot {index + 1}',
                    zorder=2)

    # Plot for every vehicle it routes
    color_index = 0
    for depot_index in range(ga.vrp_instance.n_depots):
        # Plot routes
        depot = ga.vrp_instance.depots[depot_index]
        x_pos = ga.route_data[depot_index]["x_pos"]
        y_pos = ga.route_data[depot_index]["y_pos"]
        customer_ids = ga.route_data[depot_index]["customer_ids"]

        # Initialize variables to track the current route
        start = 0
        # start loop from 1 to exclude depot
        for j in range(1, len(x_pos)):
            # Start a new route with a new color
            if x_pos[j] == depot.x and y_pos[j] == depot.y:
                # Plot the current route
                plt.plot(x_pos[start:j+1], y_pos[start:j+1], marker='o', color=colors[color_index], zorder=1)
                color_index += 1
                # Should point to depot
                start = j

        # Add customer ids and order of iteration as labels to the plot
        for j in range(len(x_pos)):
            plt.text(x_pos[j], y_pos[j], customer_ids[j], fontsize=8, ha='right')
            if j < len(x_pos) - 1:
                mid_x = (x_pos[j] + x_pos[j + 1]) / 2
                mid_y = (y_pos[j] + y_pos[j + 1]) / 2
                plt.text(mid_x, mid_y, f'{j + 1}', fontsize=8, ha='center', va='center',
                         bbox=dict(boxstyle='round', pad=0.1, edgecolor='black', facecolor='#ffffff'))

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Routes Visualization Complete')
    plt.grid(True)
    plt.legend()

    save_plot(plt, f"../results/{ga.__class__.__name__}/{ga.TIMESTAMP}", f"depot_complete_routes")

    plt.show()


def save_plot(plotting: matplotlib.pyplot, location: str, file_name: str):
    """
    Saves plots at a given location
    param: plotting - pyplot object
    param: location - destination of file to be stored
    param: file_name - name of file
    """

    os.makedirs(location, exist_ok=True)
    file_name = os.path.join(location, file_name)
    plotting.savefig(file_name)
