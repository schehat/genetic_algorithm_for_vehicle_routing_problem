import os

from matplotlib import pyplot as plt
from numpy import ndarray


def set_customer_index_list(n_depots: int, chromosome: ndarray) -> list:
    customer_index = n_depots
    customer_index_list = [customer_index]
    for depot_i in range(n_depots - 1):
        customer_index += chromosome[depot_i]
        customer_index_list.append(customer_index)

    return customer_index_list


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


def update_benchmark_afvrp(k_means, k: int, n_depots: int, n_equipment, file_path):
    """
    Creates AFVRP testing data by enhancing the MDVRPTW benchmark instance
    param: k_means - K-Means data object
    param: n_depots - number of depots
    param: n_equipment - number of equipments for the AFVRP
    """

    print(k_means.cluster_centers_)
    print(k_means.labels_)

    # Define the input and output file paths
    input_file_path = file_path
    output_file_path = file_path + "_afvrp"

    # Open the input file for reading
    with open(input_file_path, 'r') as input_file:
        # Read all lines from the input file
        lines = input_file.readlines()

    # Open the output file for writing
    with open(output_file_path, 'w') as output_file:
        for i, line in enumerate(lines):
            # Skip the lines
            if i < n_depots + 1:
                output_file.write(line)
            else:
                # Strip any leading/trailing whitespaces, add '1', and write the modified line
                modified_line = line.strip() + f' {k_means.labels_[i - (n_depots + 1)]} {(i-n_depots-1) % n_equipment} \n'
                output_file.write(modified_line)

    # Read the first element of the last line
    with open(output_file_path, 'r') as file:
        lines = file.readlines()
        if lines:
            last_line = lines[-1]
            last_line_first_element = last_line.split()[0]
            next_line_number = int(last_line_first_element) + 1

    with open(output_file_path, 'a') as output_file:
        for i in range(k):
            label_line = f'{next_line_number + i} {k_means.cluster_centers_[i][0]} {k_means.cluster_centers_[i][1]}\n'
            output_file.writelines(label_line)
