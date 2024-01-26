import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from scipy.spatial import Delaunay

from src.utility import save_plot


class Graph:

    def __init__(self, points: ndarray):
        self.points = points.copy()
        self.graph = self.__create_graph__()
        self.shortest_paths_cache = {}

    def __create_graph__(self) -> nx.classes.graph.Graph:
        # Perform Delaunay triangulation
        triangulation = Delaunay(self.points)

        graph = nx.Graph()

        # Add nodes to the graph using (x, y) coordinates as labels
        graph.add_nodes_from(tuple(coord) for coord in triangulation.points[:, :2])

        # Add edges from the Delaunay triangulation with weights as Euclidean distances
        for simplex in triangulation.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    coord1, coord2 = tuple(triangulation.points[simplex[i], :2]), tuple(
                        triangulation.points[simplex[j], :2])
                    distance = np.linalg.norm(
                        triangulation.points[simplex[i], :2] - triangulation.points[simplex[j], :2])
                    graph.add_edge(coord1, coord2, weight=distance)

        return graph

    def save_graph(self, ):
        pos = {coord: tuple(coord) for coord in self.graph.nodes}
        nx.draw(self.graph, pos, node_size=75, node_color='skyblue', font_color='black')

        # Display the plot
        plt.title('Delaunay Triangulation Graph')
        save_plot(f"{file_prefix_name}", f"uncompleted graph")

    def shortest_path_between_two_nodes(self, a: tuple, b: tuple) -> float:
        if (a, b) in self.shortest_paths_cache:
            # If the result is already cached, return it
            return self.shortest_paths_cache[(a, b)]

        shortest_path = nx.shortest_path(self.graph, source=a, target=b, weight='weight')
        # Calculate the total distance along the path
        total_distance = sum(self.graph.edges[edge]['weight'] for edge in zip(shortest_path, shortest_path[1:]))

        # Cache the result for future use
        self.shortest_paths_cache[(a, b)] = total_distance
        return total_distance
