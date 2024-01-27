from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.cluster import KMeans


class Cluster:

    def __init__(self, points: ndarray, n_depots: int, k: int, file_path):
        self.points = points.copy()
        self.n_depots = n_depots
        self.k = k
        self.file_path = file_path
        self.k_means = KMeans(n_clusters=self.k, random_state=42)
        self.k_means.fit(points)

    def plot_clusters(self):
        # Get the cluster centers and labels
        centroids = self.k_means.cluster_centers_
        labels = self.k_means.labels_

        # Separate customer and depot points
        customer_points = self.points[:-self.k]
        depot_points = self.points[-self.k:]

        # Plot the customer positions with circle markers and colors based on cluster labels
        plt.scatter(customer_points[:, 0], customer_points[:, 1], c=labels[:-self.k], cmap='viridis', label='Aufgaben',
                    marker='o')

        # Plot the depot positions with bigger circles, black edge, and colors based on cluster labels
        plt.scatter(depot_points[:, 0], depot_points[:, 1], c=labels[-self.k:], cmap='viridis', label='Depots',
                    marker='o', s=100, edgecolors='black')

        # Plot the cluster centers with red 'X' markers
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', label='Cluster Zentrum')

        plt.title('Aufgaben und Depot Clustering mit K-Means')
        plt.xlabel('X Koordinaten')
        plt.ylabel('Y Koordinaten')
        plt.legend()
        plt.show()

    def get_centroid(self):
        return self.k_means.cluster_centers_

    def get_labels(self):
        return self.k_means.labels_
