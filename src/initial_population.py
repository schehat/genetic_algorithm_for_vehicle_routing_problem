import numpy as np
from numpy import ndarray

from GA import GA
from src.plot import plot_routes
from scipy.spatial.distance import euclidean


def initial_population_random(ga: GA):
    """
    Generate random initial population
    Part 1: average assignment of customers + rest
    Part 2: random permutation
    """

    for i in range(ga.population_size):
        # Part 1: Assign average number of customers for each depot
        depot_customer_count = np.zeros(ga.vrp_instance.n_depots, dtype=int)
        total_customers_assigned = 0
        avg_customers_per_depot = ga.vrp_instance.n_customers / ga.vrp_instance.n_vehicles
        std_deviation = 1.0

        # One additional loop to guarantee all customers are assigned to depot, relevant in else block
        for depot_index in range(ga.vrp_instance.n_vehicles + 1):
            # Calculate the maximum number of customers that can be assigned to depot
            max_customers = ga.vrp_instance.n_customers - total_customers_assigned
            if max_customers < 1:
                break

            # Excluding the additional loop
            if depot_index < ga.vrp_instance.n_depots:
                # Generate a random number of customers for this depot using a Gaussian distribution
                # centered around the avg_customers_per_depot
                num_customers = int(np.random.normal(loc=avg_customers_per_depot, scale=std_deviation))
                # Ensure it's within valid bounds
                num_customers = max(1, min(max_customers, num_customers))
                depot_customer_count[depot_index] = num_customers
            else:
                # If all depots assigned and customers remain, assign the rest to random depot
                num_customers = max_customers
                depot_index = np.random.randint(ga.vrp_instance.n_depots)
                depot_customer_count[depot_index] += num_customers

            total_customers_assigned += num_customers

        # Part 2: Random order of customers for each vehicle
        order_of_customers = np.random.permutation(np.arange(1, ga.vrp_instance.n_customers + 1))

        # Combine the 2 parts to form a chromosome
        chromosome = np.concatenate((depot_customer_count, order_of_customers))
        ga.population[i]["individual"] = i
        ga.population[i]["chromosome"] = chromosome


def initial_population_grouping_savings_nnh(ga: GA):
    """
    TODO
    """

    for i in range(ga.population_size):
        # 1. grouping
        links_type = np.dtype([("customer_order", object)])
        links = np.zeros(ga.vrp_instance.n_depots, dtype=links_type)

        # Initialize the customer order lists for each depot
        links["customer_order"] = [[] for _ in range(ga.vrp_instance.n_depots)]

        assign_customers_to_closest_depots(ga, links)

        # 1. part: Calculate the number of customers for each depot
        depot_customer_count = np.array([len(link[0]) for link in links], dtype=int)

        # 2. part: Flatten the lists to get the order of customers
        order_of_customers = np.concatenate([link[0] for link in links])
        # wright_clark_savings(ga, links)
        # order_of_customers = np.concatenate([link[0] for link in links])

        # Combine the two parts to form a chromosome
        chromosome = np.concatenate((depot_customer_count, order_of_customers))

        ga.population[i]["individual"] = i
        ga.population[i]["chromosome"] = chromosome


def assign_customers_to_closest_depots(ga: GA, links: ndarray, ):
    """
    Iterate through customers and assign them to the closest depot in place
    TODO
    """
    for customer_index, customer in enumerate(ga.vrp_instance.customers):
        customer_x, customer_y = customer.x, customer.y
        closest_depot_index = None
        closest_distance = float('inf')

        for depot_index, depot in enumerate(ga.vrp_instance.depots):
            depot_x, depot_y = depot.x, depot.y
            distance = np.linalg.norm(np.array([customer_x, customer_y]) - np.array([depot_x, depot_y]))

            if distance < closest_distance:
                closest_distance = distance
                closest_depot_index = depot_index

        links["customer_order"][closest_depot_index].append(customer.id)


# def wright_clark_savings(self, depot_customer_order: list):
#     """
#         Apply Wright and Clark savings method to reorder customers in-place.
#         param: depot_customer_order - a list containing customer indices for each depot.
#         """
#     for depot_orders in depot_customer_order:
#         # Create a list of customer pairs and their corresponding savings
#         savings = []
#         for i in range(len(depot_orders)):
#             for j in range(i + 1, len(depot_orders)):
#                 customer_i, customer_j = depot_orders[i], depot_orders[j]
#                 distance_i_j = euclidean(
#                     (self.vrp_instance.customers[customer_i].x, self.vrp_instance.customers[customer_i].y),
#                     (self.vrp_instance.customers[customer_j].x, self.vrp_instance.customers[customer_j].y)
#                 )
#                 savings.append((customer_i, customer_j, distance_i_j))
#
#         # Sort the savings list by descending order of savings
#         savings.sort(key=lambda x: x[2], reverse=True)
#
#         # Reorder customers based on the savings method
#         new_order = []
#         for customer_i, customer_j, _ in savings:
#             if customer_i not in new_order and customer_j not in new_order:
#                 new_order.extend([customer_i, customer_j])
#
#         # Update the depot_orders with the new order of customers
#         depot_orders[:] = new_order
