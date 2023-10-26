import os

import matplotlib
import numpy as np

import matplotlib.pyplot as plt
from numpy import ndarray

from purpose import Purpose
from vrp import Depot, Customer


def plot_fitness(ga, width=8, height=6, interval=50):
    """
    Plot data points for minimum and average fitness values over generations at given intervals
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
    Plot the routes for a given chromosome solution
    param: ga - genetic algorithm
    param: chromosome - the best solution
    param: width and height - size of figure
    """

    colors = ["red", "cyan", "magenta", "orange"]
    route_data = []

    def collect_routes(obj, from_vehicle_i):
        """
        Add routing points and their id
        param: obj - Customer or Depot object
        param: from_vehicle_i - index from which vehicle the route is coming from
        """

        # use route_data declared above
        nonlocal route_data
        # Ensure there's an entry for the vehicle, and initialize it if not present
        while len(route_data) <= from_vehicle_i:
            route_data.append({'x_pos': [], 'y_pos': [], 'customer_ids': []})

        route_data[from_vehicle_i]['x_pos'].append(obj.x)
        route_data[from_vehicle_i]['y_pos'].append(obj.y)

        # Differentiate between Customer or Depot object
        if type(obj) is Customer:
            route_data[from_vehicle_i]['customer_ids'].append(f"C{obj.id}")
        elif type(obj) is Depot:
            # Blind label
            route_data[from_vehicle_i]['customer_ids'].append("")
        else:
            print("ERROR: unexpected behavior")

    # While decoding chromosome use collect_routes
    ga.decode_chromosome(chromosome, Purpose.PLOTTING, collect_routes)

    # Plot for every vehicle it routes
    for vehicle_index in range(ga.vrp_instance.n_vehicles):
        plt.figure(figsize=(width, height))

        # plot depots
        for index in range(ga.vrp_instance.n_depots):
            depot: Depot = ga.vrp_instance.depots[index]
            plt.scatter(depot.x, depot.y, s=150, color=colors[index], edgecolor='black', label=f'Depot {index + 1}',
                        zorder=2)

        # Plot routes
        x_pos = route_data[vehicle_index]["x_pos"]
        y_pos = route_data[vehicle_index]["y_pos"]
        customer_ids = route_data[vehicle_index]["customer_ids"]
        plt.plot(x_pos, y_pos, marker='o', color=colors[vehicle_index], zorder=1)

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
        plt.title(f'Routes Visualization Vehicle {vehicle_index + 1}')
        plt.grid(True)
        plt.legend()

        save_plot(plt, f"../results/{ga.__class__.__name__}/{ga.TIMESTAMP}", f"route_vehicle_{vehicle_index + 1}")

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
