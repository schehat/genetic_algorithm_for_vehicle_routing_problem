import random

import numpy as np
from numpy import ndarray

from GA import GA
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
    Generate initial population
    Part 1: grouping - group customers to the nearest depot
    Part 2: routing - apply savings method to create routes
    Part 3: scheduling - apply nearest neighbor heuristic to schedule the customers inside a route
    """

    for i in range(ga.population_size):
        # 1. grouping
        links_type = np.dtype([("customer_order", object)])
        links = np.zeros(ga.vrp_instance.n_depots, dtype=links_type)
        # Initialize the customer order lists for each depot
        links["customer_order"] = [[] for _ in range(ga.vrp_instance.n_depots)]
        assign_customers_to_closest_depots(ga, links)
        depot_customer_count = np.array([len(link[0]) for link in links], dtype=int)

        # 2. routing
        routing_result = []
        wright_clark_savings(ga, links, routing_result)

        # 3. grouping
        nearest_neighbor_heuristic(ga, routing_result)

        # Flatten the lists to get the order of customers
        routing_result = [customer for depot_list in routing_result for route_list in depot_list for customer in
                          route_list]

        # Combine the two parts to form a chromosome
        chromosome = np.concatenate((depot_customer_count, routing_result))

        ga.population[i]["individual"] = i
        ga.population[i]["chromosome"] = chromosome


def assign_customers_to_closest_depots(ga: GA, links: ndarray, ):
    """
    Iterate through customers and assign them to the closest depot in place
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


def wright_clark_savings(ga: GA, depot_customer_order: np.ndarray, routing_result: list):
    for depot_index, depot_orders in enumerate(depot_customer_order):
        # Create a list of customer pairs and their corresponding savings
        n_customers = len(depot_orders[0])
        savings_type = [('customer1', int), ('customer2', int), ('saving', float)]
        savings_list = np.empty((n_customers * (n_customers - 1) // 2,), dtype=savings_type)
        # Initialize the index for savings_list
        k = 0
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                depot = ga.vrp_instance.depots[depot_index]
                customer_i_id = depot_orders[0][i]
                customer_j_id = depot_orders[0][j]
                # -1 because id starts at 1 but indexing at 0
                customer_i = ga.vrp_instance.customers[customer_i_id - 1]
                customer_j = ga.vrp_instance.customers[customer_j_id - 1]

                distance_d_i = euclidean(
                    (depot.x, depot.y),
                    (customer_i.x, customer_i.y)
                )
                distance_d_j = euclidean(
                    (depot.x, depot.y),
                    (customer_j.x, customer_j.y)
                )
                distance_i_j = euclidean(
                    (customer_i.x, customer_i.y),
                    (customer_j.x, customer_j.y)
                )
                saving = distance_d_i + distance_d_j - distance_i_j

                # Update savings_list
                savings_list['customer1'][k] = i
                savings_list['customer2'][k] = j
                savings_list['saving'][k] = saving
                k += 1

        # Sort savings in descending order.
        sorted_indices = np.argsort(-savings_list['saving'])
        savings_list = savings_list[sorted_indices]

        # Initialize routes for each customer.
        routes = [[i] for i in range(n_customers)]
        cumulative_demand = [ga.vrp_instance.customers[depot_orders[0][i] - 1].demand for i in range(n_customers)]

        # Merge routes based on savings while respecting capacity constraints
        for (i, j, saving) in savings_list:
            route_i, route_j = routes[i], routes[j]

            # Because customers may already are in a route and a route has more than 2 customers
            # for more flexibility interchange from where which route is appending by separate pointers
            route_index = None
            route_to_insert = None

            # Both customers already used in a route
            if not route_i and not route_j:
                continue

            # Route_i already used then need to append route_j in the route where route_i is
            if not route_i:
                for index, route in enumerate(routes):
                    if i in route:
                        route_index = index
                        break

                # If j already where i is then continue
                if j in routes[route_index]:
                    continue

                route_to_insert = j

            # Route_j already used then need to append route_i in the route where route_j is
            elif not route_j:
                route_index = None
                for index, route in enumerate(routes):
                    if j in route:
                        route_index = index
                        break

                # If i already where j is then continue
                if i in routes[route_index]:
                    continue

                route_to_insert = i

            # normal case: appending j to i
            else:
                route_index = i
                route_to_insert = j

            if cumulative_demand[route_index] + cumulative_demand[route_to_insert] <= ga.vrp_instance.max_capacity:
                # Merge the two routes by adding the second customer's route to the first customer's route.
                routes[route_index] += routes[route_to_insert]
                # Update cumulative demand for the merged route.
                cumulative_demand[route_index] += cumulative_demand[route_to_insert]
                # Mark the second customer's route as used.
                routes[route_to_insert] = []
                cumulative_demand[route_to_insert] = 0

        # Filter out the empty routes, leaving only the non-empty merged routes.
        routes = [route for route in routes if route]
        # Replace indexes to customers by there ids
        for route in routes:
            for i in range(len(route)):
                route[i] = depot_orders[0][route[i]]
        routing_result.append(routes)


def nearest_neighbor_heuristic(ga: GA, routing_result: list):
    for depot_routes in routing_result:
        for route in depot_routes:
            if len(route) >= 2:
                # Randomly select a customer from the route
                selected_customer = random.choice(route)
                # Copy the route and start with the selected customer
                new_route = [selected_customer]
                route.remove(selected_customer)

                while route:
                    # Find the nearest neighbor to the last customer in the new route
                    last_customer = new_route[-1]
                    nearest_neighbor = min(route, key=lambda customer_id: euclidean(
                        (
                            ga.vrp_instance.customers[last_customer - 1].x,
                            ga.vrp_instance.customers[last_customer - 1].y),
                        (
                            ga.vrp_instance.customers[customer_id - 1].x,
                            ga.vrp_instance.customers[customer_id - 1].y)
                    ))
                    new_route.append(nearest_neighbor)
                    route.remove(nearest_neighbor)

                # Replace the original route with the new ordered route
                depot_routes[depot_routes.index(route)] = new_route
