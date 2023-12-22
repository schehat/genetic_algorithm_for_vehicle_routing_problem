import math
import os

import numpy as np

import matplotlib.pyplot as plt
from numpy import ndarray

from enums import Purpose
from vrp import Depot


class Plot:
    colors = ["salmon", "gold", "lightgreen", "mediumslateblue", "indianred", "orange", "limegreen", "deepskyblue",
              "yellow", "turquoise", "dodgerblue", "violet", "peru", "springgreen", "steelblue", "crimson"]
    color_i = -1
    interval = None

    def __init__(self, ga: "GA", width=8, height=6):
        self.ga = ga
        self.width = width
        self.height = height
        self.file_prefix_name = f"../BA_results/{self.ga.instance_name}"

    def plot_fitness(self):
        """
        Plot data points for minimum and average fitness values over generations at given intervals
        """

        plt.figure(figsize=(self.width, self.height))

        x = np.arange(self.ga.max_generations)
        min_fitness = self.ga.fitness_stats["min"]
        avg_fitness = self.ga.fitness_stats["avg"]

        # Plot data points at interval
        self.interval = math.floor(self.ga.generation * 0.05) if math.ceil(self.ga.generation * 0.05) > 0 else 1
        x_intervals = np.arange(self.ga.max_generations)[0:self.ga.generation+1:self.interval]
        min_fitness_intervals = min_fitness[0:self.ga.generation+1:self.interval]
        avg_fitness_intervals = avg_fitness[0:self.ga.generation+1:self.interval]

        # Last element might not include self.ga.generation data points, if not add them
        if x_intervals[-1] != self.ga.generation:
            x_intervals = np.append(x_intervals, self.ga.generation)
            min_fitness_intervals = np.append(min_fitness_intervals, min_fitness[self.ga.generation])
            # avg_fitness_intervals = np.append(avg_fitness_intervals, avg_fitness[self.ga.generation])

        plt.plot(x_intervals, min_fitness_intervals, marker='o', label='Min Fitness')
        plt.plot(x_intervals, avg_fitness_intervals, marker='o', label='Avg Fitness')

        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness over Generations')
        plt.grid(True)
        plt.legend()
        self.save_plot(f"{self.file_prefix_name}/{self.ga.TIMESTAMP}", f"fitness")
        plt.show()

    def plot_routes(self, individual: ndarray, width=12, height=10):
        """
        Plot the routes for a given chromosome solution
        param: individual - structured 3D element ["individual"]["chromosome]["fitness"]
        """

        # Decode chromosome to make route_data
        self.ga.decode_chromosome(individual, Purpose.PLOTTING)

        # Plot for every depot it routes
        for depot_i in range(self.ga.vrp_instance.n_depots):
            plt.figure(figsize=(self.width, self.height))
            self._plot_depots()
            self._plot_depot_routes(depot_i)
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title(f'Depot Route Visualization {depot_i + 1}')
            plt.grid(True)
            plt.legend()
            self.save_plot(f"{self.file_prefix_name}/{self.ga.TIMESTAMP}", f"depot_route{depot_i + 1}")
            plt.show()

        # Plot for every vehicle it routes
        plt.figure(figsize=(width, height))
        self._plot_depots()
        self.color_i = -1
        for depot_i in range(self.ga.vrp_instance.n_depots):
            self._plot_depot_routes(depot_i)

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Depot Route Visualization Complete')
        plt.grid(True)
        plt.legend()
        self.save_plot(f"{self.file_prefix_name}/{self.ga.TIMESTAMP}", f"depot_complete_routes")
        plt.show()

    def _plot_depots(self):
        for index in range(self.ga.vrp_instance.n_depots):
            depot: Depot = self.ga.vrp_instance.depots[index]
            plt.scatter(depot.x, depot.y, s=150, color=self.colors[index], edgecolor='black', label=f'Depot {index + 1}', zorder=2)

    def _plot_depot_routes(self, depot_i: int):
        depot = self.ga.vrp_instance.depots[depot_i]
        x_pos = self.ga.route_data[depot_i]["x_pos"]
        y_pos = self.ga.route_data[depot_i]["y_pos"]
        customer_ids = self.ga.route_data[depot_i]["customer_ids"]

        # Initialize variables to track the current route
        start = 0
        # start loop from 1 to exclude depot
        for j in range(1, len(x_pos)):
            # Start a new route with a new color
            if x_pos[j] == depot.x and y_pos[j] == depot.y:
                if self.color_i + 1 < len(self.colors):
                    self.color_i += 1
                else:
                    self.color_i = 0

                # Plot the current route
                plt.plot(x_pos[start:j + 1], y_pos[start:j + 1], marker='o', color=self.colors[self.color_i], zorder=1)

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

    @staticmethod
    def save_plot(location: str, file_name: str):
        """
        Saves plots at a given location
        param: plotting - pyplot object
        param: location - destination of file to be stored
        param: file_name - name of file
        """

        os.makedirs(location, exist_ok=True)
        file_name = os.path.join(location, file_name)
        plt.savefig(file_name)
