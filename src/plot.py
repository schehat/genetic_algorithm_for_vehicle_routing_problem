import os

import numpy as np

import matplotlib.pyplot as plt
from numpy import ndarray

from purpose import Purpose
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

    colors = ["red", "cyan", "magenta", "orange"]
    x_pos = []
    y_pos = []
    customer_ids = []

    def collect_routes(obj, obj_copy):
        """
        Add routing points and there id as label
        para: Customer or Depot object
        """
        nonlocal x_pos, y_pos, customer_ids
        x_pos.append(obj.x)
        y_pos.append(obj.y)
        customer_ids.append(f"C{obj.id}")

    ga.decode_chromosome(chromosome, Purpose.PLOTTING, collect_routes)

    for i in range(ga.vrp_instance.n_vehicles):
        plt.figure(figsize=(width, height))
        for index in range(ga.vrp_instance.n_depots):
            depot: Depot = ga.vrp_instance.depots[index]
            plt.scatter(depot.x, depot.y, s=150, color=colors[index], edgecolor='black', label=f'Depot {index + 1}',
                        zorder=2)

        # Plot route
        plt.plot(x_pos, y_pos, marker='o', color=colors[i], zorder=1)
        # Add customer ids as text labels
        for j in range(len(x_pos)):
            plt.text(x_pos[j], y_pos[j], f'{customer_ids[j]}', fontsize=8, ha='right')
            if j < len(x_pos) - 1:
                mid_x = (x_pos[j] + x_pos[j + 1]) / 2
                mid_y = (y_pos[j] + y_pos[j + 1]) / 2
                plt.text(mid_x, mid_y, f'{j + 1}', fontsize=8, ha='center', va='center',
                         bbox=dict(boxstyle='round', pad=0.1, edgecolor='black', facecolor='#ffffff'))

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Routes Visualization Vehicle {i + 1}')
        plt.grid(True)
        plt.legend()

        save_plot(plt, f"../results/{ga.__class__.__name__}/{ga.TIMESTAMP}", f"route_vehicle_{i + 1}")

        plt.show()
        break


def save_plot(plotting, location: str, file_name: str):
    os.makedirs(location, exist_ok=True)
    file_name = os.path.join(location, file_name)
    plotting.savefig(file_name)
