from numpy import ndarray


def set_customer_index_list(n_depots: int, chromosome: ndarray) -> list:
    customer_index = n_depots
    customer_index_list = [customer_index]
    for depot_i in range(n_depots - 1):
        customer_index += chromosome[depot_i]
        customer_index_list.append(customer_index)

    return customer_index_list
