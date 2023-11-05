import numpy as np

from GA import GA


def initial_random_population(ga: GA):
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
