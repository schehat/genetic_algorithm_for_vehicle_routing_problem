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
        plt.scatter(customer_points[:, 0], customer_points[:, 1], c=labels[:-self.k], cmap='viridis', label='Customers',
                    marker='o')

        # Plot the depot positions with bigger circles, black edge, and colors based on cluster labels
        plt.scatter(depot_points[:, 0], depot_points[:, 1], c=labels[-self.k:], cmap='viridis', label='Depots',
                    marker='o', s=100, edgecolors='black')

        # Plot the cluster centers with red 'X' markers
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', label='Cluster Centers')

        plt.title('Customer and Depot Clustering with K-Means')
        plt.xlabel('X Koordinaten')
        plt.ylabel('Y Koordinaten')
        plt.legend()
        plt.show()

    def adjust_benchmark_with_cluster(self):
        print(self.k_means.cluster_centers_)
        print(self.k_means.labels_)

        # Define the input and output file paths
        input_file_path = self.file_path
        output_file_path = self.file_path + "_afvrp"

        # Open the input file for reading
        with open(input_file_path, 'r') as input_file:
            # Read all lines from the input file
            lines = input_file.readlines()

        # Open the output file for writing
        with open(output_file_path, 'w') as output_file:
            for i, line in enumerate(lines):
                # Skip the lines
                if i < self.n_depots + 1:
                    output_file.write(line)
                else:
                    # Strip any leading/trailing whitespaces, add '1', and write the modified line
                    modified_line = line.strip() + f' {self.k_means.labels_[i - (self.n_depots + 1)]}\n'
                    output_file.write(modified_line)

        # Read the first element of the last line
        with open(output_file_path, 'r') as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1]
                last_line_first_element = last_line.split()[0]
                next_line_number = int(last_line_first_element) + 1

        with open(output_file_path, 'a') as output_file:
            for i in range(self.k):
                label_line = f'{next_line_number + i} {self.k_means.cluster_centers_[i][0]} {self.k_means.cluster_centers_[i][1]}\n'
                output_file.writelines(label_line)

    def get_centroid(self):
        return self.k_means.cluster_centers_

    def get_labels(self):
        return self.k_means.labels_
