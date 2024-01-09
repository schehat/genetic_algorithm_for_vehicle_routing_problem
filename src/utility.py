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
